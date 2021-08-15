cmake_minimum_required(VERSION 3.1)

project (multithreading)

add_executable(multithreading ./main.cpp)

target_link_libraries(
    multithreading
    -lpthread
)
