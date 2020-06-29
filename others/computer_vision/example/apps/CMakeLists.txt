
ADD_EXECUTABLE(record_images record_images.cpp)
TARGET_LINK_LIBRARIES(record_images 
    lib_static_image_properties 
    lib_time_val
)

ADD_EXECUTABLE(build_pointcloud build_pointcloud.cpp)
TARGET_LINK_LIBRARIES(build_pointcloud
    lib_cuda_common
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    cuda
)
