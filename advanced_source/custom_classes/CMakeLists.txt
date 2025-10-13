cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(infer)

find_package(Torch REQUIRED)

add_subdirectory(custom_class_project)

# Define our library target
add_executable(infer infer.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(infer "${TORCH_LIBRARIES}")
# This is where we link in our libcustom_class code, making our
# custom class available in our binary.
target_link_libraries(infer -Wl,--no-as-needed custom_class)
