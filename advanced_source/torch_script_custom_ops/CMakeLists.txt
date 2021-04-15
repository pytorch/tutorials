cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define our library target
add_library(warp_perspective SHARED op.cpp)
# Enable C++14
target_compile_features(warp_perspective PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(warp_perspective opencv_core opencv_imgproc)
