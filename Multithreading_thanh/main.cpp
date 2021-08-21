#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>
#include <mutex>

// Max length of the queues
#define MAX_QUEUE_SIZE 10

void Producer(std::queue<int> &inputQueue,
              bool &isRunning,
              std::mutex &mutex,
              int msInterval, int start) {
    // 2 producer output even and odd values
    int i = start;
    while (i < 100) {
        if (inputQueue.size() < MAX_QUEUE_SIZE) {
            mutex.lock();
            inputQueue.push(i);
            mutex.unlock();
            i += 2;
        }
        usleep(msInterval);
    }
    mutex.lock();
    isRunning = false;
    mutex.unlock();
}

void Reader(std::queue<int> &outputQueue, bool &isRunning) {
    // Reader consume output thread and print to stdout
    while (isRunning) {
        if (!outputQueue.empty()) {
            std::cout << "Output: " << outputQueue.front() << std::endl;
            outputQueue.pop();
        }
    }
}

// Implementation of Fan-in method from https://divan.dev/posts/go_concurrency_visualize/
int main() {
    std::queue<int> inputQueue;
    std::queue<int> outputQueue;
    bool isRunning1 = true;
    bool isRunning2 = true;
    bool isRunning3 = true;
    std::mutex mutex;

    // Note: BUG small sleep time cause program to freeze NOT SURE WHY
    // td::ref is required to pass queue to thread
    std::thread producer1(Producer, std::ref(inputQueue), std::ref(isRunning1), std::ref(mutex), 100, 0);
    std::thread producer2(Producer, std::ref(inputQueue), std::ref(isRunning2), std::ref(mutex), 200, 1);
    std::thread reader(Reader, std::ref(outputQueue), std::ref(isRunning3));

    // Main thread will consume producer-threads and output to reader-threads

    // Multiple pop-worker can result in non-thread-safe queue.
    while (true) {
        // If both producer are stopped
        if (!isRunning1 && !isRunning2) {
            mutex.lock();
            isRunning3 = false;
            mutex.unlock();
            break;
        }

        if (!inputQueue.empty()) {
            mutex.lock();
            int value = inputQueue.front();
            inputQueue.pop();  // This line crash without mutex
            mutex.unlock();

            if (outputQueue.size() < MAX_QUEUE_SIZE)
                outputQueue.push(value);
        }
    }

    // Wait for process to exit
    producer1.join();
    producer2.join();
    reader.join();
    return 0;
}
