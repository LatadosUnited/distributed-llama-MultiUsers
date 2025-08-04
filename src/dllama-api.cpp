#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <csignal>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include "tokenizer.hpp"
#include "app.hpp"
#include "json.hpp"
#include "api-types.hpp"
#include "nn/nn-network.hpp"

typedef unsigned int pos_t;

using json = nlohmann::json;

enum class HttpMethod {
    METHOD_GET = 0,
    METHOD_POST = 1,
    METHOD_PUT = 2,
    METHOD_DELETE = 3,
    METHOD_OPTIONS = 4,
    METHOD_UNKNOWN = 5
};

class HttpRequest {
public:
    static HttpRequest read(int serverSocket) {
        HttpRequest req(serverSocket);

        std::vector<char> httpRequest = req.readHttpRequest();
        std::string data = std::string(httpRequest.begin(), httpRequest.end());

        std::istringstream iss(data);
        std::string line;
        std::getline(iss, line);

        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        req.method = parseMethod(methodStr);
        req.path = path;

        while (std::getline(iss, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2);
                value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) || !std::isprint(c);
                }), value.end());
                req.headers[key] = value;
            }
        }

        std::getline(iss, req.body, '\0');

        if (req.body.size() > 0) {
            req.parsedJson = json::parse(req.body);
        }
        return req;
    }

    static HttpMethod parseMethod(const std::string& method) {
        if (method == "GET") return HttpMethod::METHOD_GET;
        if (method == "POST") return HttpMethod::METHOD_POST;
        if (method == "PUT") return HttpMethod::METHOD_PUT;
        if (method == "DELETE") return HttpMethod::METHOD_DELETE;
        if (method == "OPTIONS") return HttpMethod::METHOD_OPTIONS;
        return HttpMethod::METHOD_UNKNOWN;
    }

private:
    int serverSocket;
public:
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    json parsedJson;
    HttpMethod method;

    HttpRequest(int serverSocket) {
        this->serverSocket = serverSocket;
    }

    std::vector<char> readHttpRequest() {
        std::string httpRequest;
        char buffer[1024 * 64];
        ssize_t bytesRead;

        std::string headerData;
        size_t headerEnd;
        bool headerDone = false;
        std::string extraReadPastHeader;
        while (!headerDone) {
            bytesRead = recv(serverSocket, buffer, sizeof(buffer) - 1, 0);
            if (bytesRead <= 0) {
                throw std::runtime_error("Error while reading headers from socket");
            }
            buffer[bytesRead] = '\0';
            headerData.append(buffer);

            headerEnd = headerData.find("\r\n\r\n");
            if (headerEnd != std::string::npos) {
                headerDone = true;
                if (headerEnd < headerData.size()-4) {
                    extraReadPastHeader = headerData.substr(headerEnd+4);
                }
            }
        }

        httpRequest.append(headerData);

        std::istringstream headerStream(headerData);
        std::string line;
        ssize_t contentLength = 0;
        while (std::getline(headerStream, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2);
                if (key == "Content-Length") {
                    try {
                      contentLength = std::stoi(value);
                    } catch (const std::invalid_argument& e) {
                      throw std::runtime_error("Bad Content-Length header - not a number");
                    }
                    break;
                }
            }
        }

        if (contentLength > 0) {
            if (extraReadPastHeader.size() > static_cast<size_t>(contentLength)) {
                throw std::runtime_error("Received more body data than Content-Length header said");
            }
            contentLength -= extraReadPastHeader.size();

            std::vector<char> body(contentLength);
            ssize_t totalRead = 0;
            while (totalRead < contentLength) {
                bytesRead = recv(serverSocket, body.data() + totalRead, contentLength - totalRead, 0);
                if (bytesRead <= 0) {
                    throw std::runtime_error("Error while reading body from socket");
                }
                totalRead += bytesRead;
            }
            if (body.size() > 0) {
              httpRequest.append(body.data(), contentLength);
            }
        }

        return std::vector<char>(httpRequest.begin(), httpRequest.end());
    }

    std::string getMethod() {
        if (method == HttpMethod::METHOD_GET) return "GET";
        if (method == HttpMethod::METHOD_POST) return "POST";
        if (method == HttpMethod::METHOD_PUT) return "PUT";
        if (method == HttpMethod::METHOD_DELETE) return "DELETE";
        if (method == HttpMethod::METHOD_OPTIONS) return "OPTIONS";
        return "UNKNOWN";
    }
 
    void writeCors() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 204 No Content\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE\r\n"
            << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            << "Connection: close\r\n"
            << "\r\n";
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeNotFound() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 404 Not Found\r\n"
            << "Connection: close\r\n"
            << "Content-Length: 9\r\n"
            << "\r\n"
            << "Not Found";
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeJson(std::string json) {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Content-Length: " << json.length() << "\r\n\r\n" << json;
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }
};

struct Route {
    std::string path;
    HttpMethod method;
    std::function<void(HttpRequest&)> handler;
};

class Router {
public:
    static void resolve(HttpRequest& request, std::vector<Route>& routes) {
        if (request.method == HttpMethod::METHOD_OPTIONS) {
            request.writeCors();
            return;
        }
        for (const auto& route : routes) {
            if (request.method == route.method && request.path == route.path) {
                route.handler(request);
                return;
            }
        }
        request.writeNotFound();
    }
};

class ApiServer {
private:
    AppInferenceContext *context;
    AppCliArgs *args;

public:
    ApiServer(AppInferenceContext *context) {
        this->context = context;
        this->args = context->args;
    }

    void complete(HttpRequest& request) {
        InferenceParams params = parseRequest(request);

        std::string prompt_text = "";
        if (!params.messages.empty()) {
            for(const auto& msg : params.messages) {
                prompt_text += msg.role + ": " + msg.content + "\n";
            }
        }

        if (prompt_text.empty()) {
            request.writeJson("{\"error\": \"Prompt is empty.\"}");
            return;
        }

        std::promise<std::string> promise;
        std::future<std::string> future = promise.get_future();

        Request req;
        static std::atomic<int> request_id_counter(0);
        req.id = request_id_counter++;
        req.prompt = prompt_text;
        req.promise = std::move(promise);

        if (request.parsedJson.contains("max_tokens")) {
            req.max_tokens = request.parsedJson["max_tokens"].get<int>();
        }

        context->request_queue.push(std::move(req));

        std::string generated_text = future.get();

        json response_json;
        response_json["generated_text"] = generated_text;
        request.writeJson(response_json.dump());

        printf("🔶\n");
        fflush(stdout);
    }

private:
    InferenceParams parseRequest(HttpRequest& request) {
        InferenceParams params;
        params.temperature = args->temperature;
        params.top_p = args->topp;
        params.seed = args->seed;
        params.stream = false;
        params.messages = parseChatMessages(request.parsedJson["messages"]);
        params.max_tokens = -1;

        if (request.parsedJson.contains("stream")) {
            params.stream = request.parsedJson["stream"].get<bool>();
        }
        if (request.parsedJson.contains("temperature")) {
            params.temperature = request.parsedJson["temperature"].template get<float>();
        }
        if (request.parsedJson.contains("seed")) {
            params.seed = request.parsedJson["seed"].template get<unsigned long long>();
        }
        if (request.parsedJson.contains("max_tokens")) {
            params.max_tokens = request.parsedJson["max_tokens"].template get<int>();
        }
        return params;
    }
};

void handleCompletionsRequest(HttpRequest& request, ApiServer *api) {
    api->complete(request);
}

void handleModelsRequest(HttpRequest& request, const char* modelPath) {
    std::string path(modelPath);
    size_t pos = path.find_last_of("/\\");
    std::string modelName = (pos == std::string::npos) ? path : path.substr(pos + 1);

    Model model(modelName);
    ModelList list(model);
    std::string response = ((json)list).dump();
    request.writeJson(response);
}

static void server(AppInferenceContext *context) {
    int serverSocket = createServerSocket(context->args->port);

    ApiServer api(context);

    printf("Server URL: http://127.0.0.1:%d/v1/\n", context->args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, &api)
        },
        {
            "/v1/models",
            HttpMethod::METHOD_GET,
            std::bind(&handleModelsRequest, std::placeholders::_1, context->args->modelPath)
        }
    };

    while (true) {
        try {
            int clientSocket = acceptSocket(serverSocket);
            HttpRequest request = HttpRequest::read(clientSocket);
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());
            Router::resolve(request, routes);
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
        } catch (std::exception& ex) {
            printf("Socket error: %s\n", ex.what());
        }
    }

    closeServerSocket(serverSocket);
}

#ifdef _WIN32
    #define EXECUTABLE_NAME "dllama-api.exe"
#else
    #define EXECUTABLE_NAME "dllama-api"
#endif

void usage() {
    fprintf(stderr, "Usage: %s {--model <path>} {--tokenizer <path>} [--port <p>]\n", EXECUTABLE_NAME);
    fprintf(stderr, "        [--buffer-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--max-seq-len <max>]\n");
    fprintf(stderr, "        [--nthreads <n>]\n");
    fprintf(stderr, "        [--workers <ip:port> ...]\n");
    fprintf(stderr, "        [--temperature <temp>]\n");
    fprintf(stderr, "        [--topp <t>]\n");
    fprintf(stderr, "        [--seed <s>]\n");
    fflush(stderr);
}

int main(int argc, char *argv[]) {
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, false);
        if (args.help) {
            usage();
        } else {
            runInferenceApp(&args, server);
        }
    } catch (std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}