cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_class SHARED class.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(custom_class "${TORCH_LIBRARIES}")
