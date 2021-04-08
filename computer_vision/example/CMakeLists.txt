CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)


option(BUILD_EXAMPLES "Build examples and test" ON)

# configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRINGS "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
ENDIF()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")
else()
    message("debug type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
    set(CUDA_DEBUG_SYNC_CHECK "True")
endif()

find_package(k4a REQUIRED)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#set(OpenCV_LIB_DIR ${OpenCV_INSTALL_PATH}/lib)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "    include dir：${OpenCV_INCLUDE_DIRS}")
#message(STATUS "    lib dir：${OpenCV_LIB_DIR}")
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
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
include_directories(${CUDA_INCLUDE_DIRS})

# boost
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_subdirectory(cpp)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
