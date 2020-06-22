CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)


option(BUILD_DEVS "Build devs: some codes for testing new features/functions" OFF)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")

FIND_PACKAGE(k4a REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/common)
add_subdirectory(${PROJECT_SOURCE_DIR}/apps)

if (BUILD_DEVS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/devs)
endif()
