add_executable(test_tvl1 
  test_tvl1.cpp
  motion_to_color.cpp
)
target_link_libraries(test_tvl1
  lib_dataset
  ${OpenCV_LIBS}
)

add_executable(test_farneback 
  test_farneback.cpp
  motion_to_color.cpp
)
target_link_libraries(test_farneback
  lib_dataset
  ${OpenCV_LIBS}
)