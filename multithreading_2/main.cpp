#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>
#include <mutex>
#include <cstdlib>

// Max length of the queues
#define MAX_QUEUE_SIZE 10
#define DATA_PRODUCER_1 1
#define DATA_PRODUCER_2 2

 enum dataStatus {
    DATA_UNCLASSIFIED = 0,
    DATA_GOOD         = 1,
    DATA_NOTGOOD      = 2
};

struct dataInfo {
    int id;
    int data;
    dataStatus status;
};

void Data_Producer(std::queue<dataInfo> &batchInferQueue, std::queue<dataInfo> &classifiedData,
              std::mutex &mutex,
              int msInterval, int dataProducerId) {

    dataInfo producerData;
    int i = 0;

    while (i < 10) {
        i++;
        producerData.id = dataProducerId;
        producerData.data = rand() % 100; // random a value from 0 to 100, mock data
        producerData.status = DATA_UNCLASSIFIED;

        if (batchInferQueue.size() < MAX_QUEUE_SIZE) {
            mutex.lock();
            batchInferQueue.push(producerData);
            mutex.unlock();
        }
        usleep(msInterval);
    }
    usleep(msInterval*2);
    // print out the classified data
    while (!classifiedData.empty()) {
        mutex.lock();
        dataInfo output = classifiedData.front();
        classifiedData.pop();
        std::string status = (output.status == DATA_GOOD)?"GOOD":"NOT_GOOD";
        std::cout << "Producer " << dataProducerId << " :\n\t data: " << output.data <<"\n\t status: " << status << std::endl;
        mutex.unlock();
    }
}

int main() {
    std::queue<dataInfo> batchInferQueue;
    std::queue<dataInfo> classifiedData_1;
    std::queue<dataInfo> classifiedData_2;

    std::mutex mutex;

    std::thread Data_Producer_1(Data_Producer, std::ref(batchInferQueue), std::ref(classifiedData_1),
                                std::ref(mutex), 100, DATA_PRODUCER_1);
    std::thread Data_Producer_2(Data_Producer, std::ref(batchInferQueue), std::ref(classifiedData_2),
                                std::ref(mutex), 100, DATA_PRODUCER_2);

    // Main thread will consume Data_Producer and return the result to threads
    // classify the data which odd is NOT GOOD and even is GOOD
    while (true) {

        if (!batchInferQueue.empty()) {
            mutex.lock();
            dataInfo value = batchInferQueue.front();
            batchInferQueue.pop();
            if (value.data % 2 == 0) {
                value.status = DATA_GOOD;
            }
            else {
                value.status = DATA_NOTGOOD;
            }
            std::string status = (value.status == DATA_GOOD)?"GOOD":"NOT_GOOD";
            std::cout << "processing data for Producer " << value.id <<" value: " << value.data <<"\n" << "dataStatus: " << status << std::endl;
            mutex.unlock();

            if (value.id == DATA_PRODUCER_1) {
                mutex.unlock();
                classifiedData_1.push(value);
                mutex.unlock();
            }
            else {
                mutex.unlock();
                classifiedData_2.push(value);
                mutex.unlock();
            }
        }
    }

    // Wait for process to exit
    Data_Producer_1.join();
    Data_Producer_2.join();
    return 0;
}
