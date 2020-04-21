CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)


#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")

FIND_PACKAGE(k4a REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


ADD_EXECUTABLE(get_images_bak get_images_bak.cpp)
TARGET_LINK_LIBRARIES(get_images_bak k4a::k4a ${OpenCV_LIBS})

ADD_EXECUTABLE(get_images get_images.cpp StaticImageProperties.cpp Util.cpp)
TARGET_LINK_LIBRARIES(get_images k4a::k4a ${OpenCV_LIBS})

ADD_EXECUTABLE(transform_images transform_images.cpp)
TARGET_LINK_LIBRARIES(transform_images k4a::k4a ${OpenCV_LIBS})


ADD_EXECUTABLE(undistort_images undistort_images.cpp)
TARGET_LINK_LIBRARIES(undistort_images k4a::k4a ${OpenCV_LIBS})
