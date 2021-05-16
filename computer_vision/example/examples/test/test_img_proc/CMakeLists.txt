ADD_EXECUTABLE(test_pyramid test_pyramid.cpp)
TARGET_LINK_LIBRARIES(test_pyramid lib_imgproc)

add_executable(test_img_proc test_img_proc.cpp)
target_link_libraries(test_img_proc
    lib_dataset
    lib_imgproc
    lib_visualize
    lib_common
)

add_executable(test_icp test_icp.cpp)
target_link_libraries(test_icp
    lib_dataset
    lib_imgproc
    lib_visualize
    lib_common
)