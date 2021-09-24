cmake_minimum_required(VERSION 3.1)

project (multithreading_2)

add_executable(multithreading_2 ./main.cpp)

target_link_libraries(
        multithreading_2
        -lpthread
)
