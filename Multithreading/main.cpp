#include <iostream>
#include <thread>
using namespace std;

void Show_Something(string str) {
    while(1) {
        std::cout << str << "\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    std::thread thread_1, thread_2;

    std::cout << "Start MultiThreading\n";

    thread_1 = std::thread(Show_Something, "Thread 1");
    thread_2 = std::thread(Show_Something, "Thread 2");

    thread_1.join();
    thread_2.join();

    return 0;
}