cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(libtorch_examples)

find_package(Torch 1.8.1 EXACT PATHS /opt/minus/libs/libtorch NO_DEFAULT_PATH)
find_package(OpenCV 4.5.2 EXACT PATHS /opt/minus/libs/opencv-4.5.2 NO_DEFAULT_PATH)


add_executable(libtorch_examples ./main.cpp
               include/cifar10.h src/cifar10.cpp
               include/convnet.h src/convnet.cpp
               include/transform.h src/transform.cpp
              )
include_directories( ${Torch_INCLUDE_DIRS}
                     ${OpenCV_INCLUDE_DIRS}
                   )
target_link_libraries(libtorch_examples ${TORCH_LIBRARIES} ${OpenCV_LIBS})