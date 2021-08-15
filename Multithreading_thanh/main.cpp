#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>
#include <mutex>

// Max length of the queues
#define MAX_QUEUE_SIZE 10

[[noreturn]] void Producer(std::queue<int> &inputQueue,
                           std::mutex &mutex,
                           int msInterval, int start) {
    // 2 producer output even and odd values
    int i = start;
    while (true) {
        if (inputQueue.size() < MAX_QUEUE_SIZE) {
            mutex.lock();
            inputQueue.push(i);
            mutex.unlock();
            i += 2;
        }
        usleep(msInterval);
    }
}

[[noreturn]] void Reader(std::queue<int> &outputQueue) {
    // Reader consume output thread and print to stdout
    while (true) {
        if (!outputQueue.empty()) {
            std::cout << "Output: " << outputQueue.front() << std::endl;
            outputQueue.pop();
        }
    }
}

// Implementation of Fan-in method from https://divan.dev/posts/go_concurrency_visualize/
[[noreturn]] int main() {
    std::queue<int> inputQueue;
    std::queue<int> outputQueue;
    std::mutex mutex;

    // Note: BUG small sleep time cause program to freeze NOT SURE WHY
    // td::ref is required to pass queue to thread
    std::thread producer1(Producer, std::ref(inputQueue), std::ref(mutex), 100, 0);
    std::thread producer2(Producer, std::ref(inputQueue), std::ref(mutex), 200, 1);
    std::thread reader(Reader, std::ref(outputQueue));

    // Main thread will consume producer-threads and output to reader-threads

    // Multiple pop-worker can result in non-thread-safe queue.
    while (true) {
        if (!inputQueue.empty()) {
            mutex.lock();
            int value = inputQueue.front();
            inputQueue.pop();  // This line crash without mutex
            mutex.unlock();

            if (outputQueue.size() < MAX_QUEUE_SIZE)
                outputQueue.push(value);
        }
    }

    // // Wait for process to exit
    // producer1.join();
    // producer2.join();
    // reader.join();
    // return 0;
}
