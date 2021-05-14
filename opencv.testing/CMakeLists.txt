cmake_minimum_required(VERSION 3.1)
project (hello_opencv)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hello_opencv ./main.cpp)