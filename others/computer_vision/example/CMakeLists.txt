CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)


option(BUILD_TEST "Build devs: some codes for testing new features/functions" ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -std=c++11")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")

find_package(k4a REQUIRED)

# opencv
find_package(OpenCV REQUIRED)
set(OpenCV_LIB_DIR ${OpenCV_INSTALL_PATH}/lib)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "    include dir：${OpenCV_INCLUDE_DIRS}")
message(STATUS "    lib dir：${OpenCV_LIB_DIR}")
message(STATUS "    libs：${OpenCV_LIBS}")

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "    include dir: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    libs：${CUDA_LIBRARIES}")
# nvcc flags
set(CUDA_NVCC_FLAGS -G;-g;-std=c++11) 
#set(CUDA_NVCC_FLAGS -std=c++11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
include_directories(${CUDA_INCLUDE_DIRS})

# boost
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/common)
add_subdirectory(${PROJECT_SOURCE_DIR}/apps)
add_subdirectory(${PROJECT_SOURCE_DIR}/dataset)
add_subdirectory(${PROJECT_SOURCE_DIR}/visualize)

if (BUILD_TEST)
    add_subdirectory(${PROJECT_SOURCE_DIR}/test)
endif()
