cmake_minimum_required(VERSION 3.1)

project (ReadImageLib)

SET(CMAKE_INSTALL_PREFIX "../installed_lib")

find_package(OpenCV REQUIRED)

#Generate the shared library from the library sources
add_library(ReadImageLib SHARED ./read_image_lib/Read_Image.cpp)

target_include_directories(ReadImageLib PUBLIC ${PROJECT_SOURCE_DIR}/read_image_lib)

add_executable(ReadImageExec ./main.cpp)

message(STATUS ${PROJECT_SOURCE_DIR})
message(${CMAKE_INSTALL_PREFIX})

include_directories(    ${OpenCV_INCLUDE_DIRS}
                   )

# Link libraries, that means link implementation of the header files
target_link_libraries( ReadImageExec
        PRIVATE ReadImageLib
        ${OpenCV_LIBS}
)

############################################################
# Install
############################################################

# Binaries
install (TARGETS ReadImageExec
    DESTINATION bin)

# Library
install (TARGETS ReadImageLib
    LIBRARY DESTINATION lib)

# Header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/read_image_lib/
    DESTINATION include)