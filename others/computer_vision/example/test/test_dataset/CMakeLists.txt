add_executable(test_nyuv2_dataset test_nyuv2_dataset.cpp )
target_link_libraries(test_nyuv2_dataset 
    lib_dataset
)

add_executable(test_azure_kinect_dataset test_azure_kinect_dataset.cpp)
target_link_libraries(test_azure_kinect_dataset 
    lib_dataset
)

add_executable(test_camera_params test_camera_params.cpp )