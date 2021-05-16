add_executable(test_dataset test_dataset.cpp)
target_link_libraries(test_dataset 
    lib_dataset
)

add_executable(test_camera_params test_camera_params.cpp )