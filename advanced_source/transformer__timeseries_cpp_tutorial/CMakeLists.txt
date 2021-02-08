cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(transformer_ts transformer_timeseries.cpp)
target_link_libraries(transformer_ts "${TORCH_LIBRARIES}")
set_property(TARGET transformer_ts PROPERTY CXX_STANDARD 14)
