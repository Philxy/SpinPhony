cmake_minimum_required(VERSION 3.18)

project(SpinPhony LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Add all .cpp and .cu files here (headers are found automatically)
add_executable(spinphony_run
    src/main.cpp
    src/cpu_logic.cpp
    src/gpu_logic.cu
)