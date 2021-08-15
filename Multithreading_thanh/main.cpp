#include "bits/stdc++.h"
#include <thread>
#include <queue>
#include <unistd.h>

using namespace std;

#define MAX_VALUE 10
#define MAX_QUEUE_SIZE 10

void Producer(queue<int> &inputQueue, int msInterval, int start) {
    int i = start;
    while (i < MAX_VALUE) {
        if (inputQueue.size() < MAX_QUEUE_SIZE) {
            inputQueue.push(i);
            i += 2;
        }
        usleep(msInterval);
    }
}

void Reader(queue<int> &outputQueue) {
    while (!outputQueue.empty()) {
        cout << "Output: " << outputQueue.front() << endl;
        outputQueue.pop();
    }
}

// Implementation of Fan-in method from https://divan.dev/posts/go_concurrency_visualize/
int main() {
    queue<int> inputQueue;
    queue<int> outputQueue;

    // Multiple pop-worker can result in non-thread-safe queue.
    // Hence, use single output queue only
    std::thread producer1(Producer, std::ref(inputQueue), 100, 0);
    std::thread producer2(Producer, std::ref(inputQueue), 250, 1);
    std::thread reader(Reader, std::ref(outputQueue));

    while (!inputQueue.empty()) {
        int value = inputQueue.front();
        inputQueue.pop();

        outputQueue.push(value);
    }

    // Wait for process to exit
    producer1.join();
    producer2.join();
    reader.join();

    return 0;
}
