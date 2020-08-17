
ADD_EXECUTABLE(test_get_images test_get_images.cpp)
TARGET_LINK_LIBRARIES(test_get_images lib_static_image_properties)

ADD_EXECUTABLE(test_transform_images test_transform_images.cpp)
TARGET_LINK_LIBRARIES(test_transform_images k4a::k4a ${OpenCV_LIBS})


ADD_EXECUTABLE(test_undistort_images test_undistort_images.cpp)
TARGET_LINK_LIBRARIES(test_undistort_images k4a::k4a ${OpenCV_LIBS})

ADD_EXECUTABLE(test_colorize_depth_image test_colorize_depth_image.cpp)
TARGET_LINK_LIBRARIES(test_colorize_depth_image lib_static_image_properties)

ADD_EXECUTABLE(test_pyramid test_pyramid.cpp)
TARGET_LINK_LIBRARIES(test_pyramid lib_cuda_common ${OpenCV_LIBS})

add_subdirectory(test_dataset)

#add_subdirectory(test_visualize)