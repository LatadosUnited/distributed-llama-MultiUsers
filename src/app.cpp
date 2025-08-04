#include "app.hpp"
#include "Request.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif

// Forward declaration for the inference loop
void inference_loop(AppInferenceContext* context);

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\000';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i += count - 1;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution), -1, -1));
    }
    return devices;
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network; // May be nullptr!
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::forward() {
    if (network != nullptr) 
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    executor->forward();
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    }
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->positionPipe = (float *)execution->pipes[0];
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    if (controlPacket.batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    for (NnUint i = 0; i < controlPacket.batchSize; i++)
        positionPipe[i] = (float)(controlPacket.position + i);
    execution->setBatchSize(controlPacket.batchSize);
    return true;
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;

    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    if (nNodes > header.nKvHeads)
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");

    Tokenizer tokenizer(args->tokenizerPath);
    if (tokenizer.vocabSize != header.vocabSize)
        throw std::runtime_error("Tokenizer vocab size does not match the model vocab size");

    Sampler sampler(header.vocabSize, args->temperature, args->topp, args->seed);

    LlmNet net = buildLlmNet(&header, nNodes, args->nBatches);
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    printLlmHeader(&header);
    printNodeRequiredMemory(&net.netConfig, rootNodeConfig);

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution);
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    NnRootWeightLoader weightLoader(&executor, network, nNodes);
    loadLlmNetWeight(args->modelPath, &net, &weightLoader);

    RootLlmInference inference(&net, &execution, &executor, network);

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.tokenizer = &tokenizer;
    context.sampler = &sampler;
    context.network = network;
    context.executor = &executor;

    // Start the inference loop in a new thread
    context.inference_thread = std::thread(inference_loop, &context);

    handler(&context);

    // Proper shutdown would require signaling the inference_thread to stop
    // and then joining it. For now, we'll just let it run.
    if(context.inference_thread.joinable()) {
        // For a clean shutdown, you'd signal a stop and then join.
        // context.stop_signal.set_value(); // Example signal
        context.inference_thread.join();
    }

    inference.finish();
}

void inference_loop(AppInferenceContext* context) {
    std::vector<Request> active_requests;
    
    while (true) { // Main loop
        // 1. Add new requests from the queue
        while (!context->request_queue.is_empty()) {
            active_requests.push_back(context->request_queue.pop());
        }

        if (active_requests.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 2. Prepare batch for inference
        std::vector<Request*> batch;
        for (auto& req : active_requests) {
            if (req.state != DONE) {
                batch.push_back(&req);
            }
        }

        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        context->inference->setBatchSize(batch.size());

        // 3. Process batch
        for (size_t i = 0; i < batch.size(); ++i) {
            Request* req = batch[i];

            if (req->state == PROMPT_PROCESSING) {
                // On the first run, process the prompt
                int num_prompt_tokens = 0;
                std::vector<int> prompt_tokens(req->prompt.length() + 3); // allocate space
                context->tokenizer->encode((char*)req->prompt.c_str(), prompt_tokens.data(), &num_prompt_tokens, true, false);
                req->tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + num_prompt_tokens);

                if (req->tokens.empty()) {
                    req->state = DONE;
                    req->promise.set_value("[ERROR] Empty prompt after tokenization.");
                    continue;
                }
                req->next_token_pos = 0;
                context->inference->setToken(i, req->tokens[req->next_token_pos]);
                context->inference->setPosition(req->next_token_pos);
                req->next_token_pos++;
            } else { // GENERATING
                // For subsequent tokens
                context->inference->setToken(i, req->tokens.back());
                context->inference->setPosition(req->next_token_pos - 1);
            }
        }

        // 4. Run model inference
        context->inference->forward();

        // 5. Sample next token and handle state changes
        for (size_t i = 0; i < batch.size(); ++i) {
            Request* req = batch[i];
            if (req->state == DONE) continue;

            int next_token = context->sampler->sample(context->inference->logitsPipe + i * context->header->vocabSize);
            req->tokens.push_back(next_token);
            req->generated_text += context->tokenizer->vocab[next_token];
            req->generated_token_count++;

            if (req->state == PROMPT_PROCESSING) {
                req->state = GENERATING;
            }

            // Check for stop conditions
            if (context->tokenizer->isEos(next_token) || 
                (req->max_tokens > 0 && req->generated_token_count >= req->max_tokens)) {
                req->state = DONE;
                req->promise.set_value(req->generated_text);
            }
            req->next_token_pos++;
        }

        // 6. Remove completed requests
        active_requests.erase(
            std::remove_if(active_requests.begin(), active_requests.end(),
                           [](const Request& req) { return req.state == DONE; }),
            active_requests.end());
    }
}


void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution);
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig);
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        NnWorkerWeightReader weightReader(&executor, network);
        weightReader.read();

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnReadNetworkException &e) {
                printf("Read network exception: %s\n", e.message);
                break;
            } catch (const NnWriteNetworkException &e) {
                printf("Write network exception: %s\n", e.message);
                break;
            }
        }
    }
}