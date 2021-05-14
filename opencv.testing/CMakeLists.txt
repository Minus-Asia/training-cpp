cmake_minimum_required(VERSION 3.1)

project (hello_opencv)

find_package(OpenCV REQUIRED)

add_executable(hello_opencv ./main.cpp)

# include opencv header files
include_directories(${OpenCV_INCLUDE_DIRS})

# Print out opencv header files
message(STATUS ${OpenCV_LIBS})
# Print out opencv sub libraries
message(STATUS ${OpenCV_INCLUDE_DIRS})

# Link libraries, that means link implementation of the header files
target_link_libraries(
        hello_opencv
        ${OpenCV_LIBS}
)

