cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dispatcher)

find_package(Torch REQUIRED)

add_library(dispatcher SHARED op.cpp)
target_compile_features(dispatcher PRIVATE cxx_std_14)
target_link_libraries(dispatcher "${TORCH_LIBRARIES}")
