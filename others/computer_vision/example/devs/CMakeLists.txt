
ADD_EXECUTABLE(get_images get_images.cpp)
TARGET_LINK_LIBRARIES(get_images lib_static_image_properties)

ADD_EXECUTABLE(transform_images transform_images.cpp)
TARGET_LINK_LIBRARIES(transform_images k4a::k4a ${OpenCV_LIBS})


ADD_EXECUTABLE(undistort_images undistort_images.cpp)
TARGET_LINK_LIBRARIES(undistort_images k4a::k4a ${OpenCV_LIBS})

ADD_EXECUTABLE(colorize_depth_image colorize_depth_image.cpp)
TARGET_LINK_LIBRARIES(colorize_depth_image lib_static_image_properties)

ADD_EXECUTABLE(test_pyramid test_pyramid.cpp)
TARGET_LINK_LIBRARIES(test_pyramid lib_cuda_common ${OpenCV_LIBS})