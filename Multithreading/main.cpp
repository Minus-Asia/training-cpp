#include <iostream>
#include <thread>
#include <semaphore.h>

using namespace std;
sem_t pingsem;

void Player(string playerName, int *ball) {
    while(1) {
        sem_wait(&pingsem);
        (*ball)++;
        std::cout << playerName << ":"<< *ball << "\n" << std::endl;
        sem_post(&pingsem);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int main() {
    sem_destroy(&pingsem);
    sem_init(&pingsem, 0, 1);

    std::cout << "Start Ping Pong\n";
    int ball = 0;
    std::thread Player1(Player, "Player 1", &ball);
    std::thread Player2(Player, "Player 2", &ball);

    Player1.join();
    Player2.join();
    return 0;
}