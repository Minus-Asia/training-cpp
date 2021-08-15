cmake_minimum_required(VERSION 3.13)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(cpp main.cpp)

target_link_libraries(cpp pthread)
