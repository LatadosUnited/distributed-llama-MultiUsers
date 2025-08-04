#ifndef REQUEST_HPP
#define REQUEST_HPP

#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

enum RequestState {
    // The prompt is being processed
    PROMPT_PROCESSING,
    // New tokens are being generated
    GENERATING,
    // The request is finished
    DONE
};

// Holds all the information for a single user request
struct Request {
    int id;
    std::string prompt;
    std::promise<std::string> promise;
    std::string generated_text;
    
    // State for the generation process
    RequestState state = PROMPT_PROCESSING;
    std::vector<int> tokens;
    size_t next_token_pos = 0; // to track position in the KV cache

    // Generation parameters
    int max_tokens = 128;
    bool stop_on_eos = true;
    int generated_token_count = 0;
};

// A thread-safe queue to hold incoming requests
class RequestQueue {
public:
    void push(Request&& req) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.push(std::move(req));
        cv.notify_one();
    }

    Request pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !queue.empty(); });
        Request req = std::move(queue.front());
        queue.pop();
        return req;
    }

    bool is_empty() {
        std::unique_lock<std::mutex> lock(mtx);
        return queue.empty();
    }

private:
    std::queue<Request> queue;
    std::mutex mtx;
    std::condition_variable cv;
};

#endif // REQUEST_HPP
