cmake_minimum_required(VERSION 3.1)

project (share_lib)

#Generate the shared library from the library sources
add_library(share_lib SHARED ./shared_lib/Hello_Name.cpp)

target_include_directories(share_lib PUBLIC ${PROJECT_SOURCE_DIR}/shared_lib)

add_executable(share_lib_exec ./main.cpp)

message(STATUS ${PROJECT_SOURCE_DIR})

# Link libraries, that means link implementation of the header files
target_link_libraries( share_lib_exec
        PRIVATE share_lib
)

