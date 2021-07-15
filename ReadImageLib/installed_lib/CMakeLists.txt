cmake_minimum_required(VERSION 3.1)

project (ReadImageLib)
find_package(OpenCV REQUIRED)
add_executable(ReadImageExec ./main.cpp)

message(STATUS ${PROJECT_SOURCE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                   )

# Link libraries, that means link implementation of the header files
target_link_libraries( ReadImageExec
        PRIVATE
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libReadImageLib.so
)
