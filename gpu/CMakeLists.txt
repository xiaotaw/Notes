cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(m_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 9 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_ARCH 61)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")

cuda_add_executable(mtest test.cu)
