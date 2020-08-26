
ADD_EXECUTABLE(record_images record_images.cpp)
TARGET_LINK_LIBRARIES(record_images 
    lib_imgproc
    lib_time
)

ADD_EXECUTABLE(build_pointcloud build_pointcloud.cpp)
TARGET_LINK_LIBRARIES(build_pointcloud
    lib_time
    lib_imgproc
    ${PCL_LIBRARIES}
)

ADD_EXECUTABLE(build_pointcloud_k4a build_pointcloud_k4a.cpp)
TARGET_LINK_LIBRARIES(build_pointcloud_k4a
    lib_time
    lib_imgproc
    ${PCL_LIBRARIES}
)

