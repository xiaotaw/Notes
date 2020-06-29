CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)


option(BUILD_DEVS "Build devs: some codes for testing new features/functions" OFF)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -std=c++11")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")

FIND_PACKAGE(k4a REQUIRED)

FIND_PACKAGE(OpenCV REQUIRED)
set(OpenCV_LIB_DIR ${OpenCV_INSTALL_PATH}/lib)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "    include dir：${OpenCV_INCLUDE_DIRS}")
message(STATUS "    lib dir：${OpenCV_LIB_DIR}")
message(STATUS "    libs：${OpenCV_LIBS}")


find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})


FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "    include dir: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    libs：${CUDA_LIBRARIES}")
# nvcc flags
set(CUDA_NVCC_FLAGS -G;-g;-std=c++11) 
#set(CUDA_NVCC_FLAGS -std=c++11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
include_directories(${CUDA_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/common)
add_subdirectory(${PROJECT_SOURCE_DIR}/apps)

if (BUILD_DEVS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/devs)
endif()
