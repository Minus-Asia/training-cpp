cmake_minimum_required(VERSION 3.1)

project (CropImageFromContour)

find_package(OpenCV REQUIRED)

add_executable(CropImageFromContour ./main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(CropImageFromContour ${OpenCV_LIBS})