add_executable(test_glfw_glad test_glfw_glad.cpp)
target_link_libraries(test_glfw_glad 
    lib_visualize
)

add_executable(test_triangle test_triangle.cpp)
target_link_libraries(test_triangle 
    lib_visualize
)

add_executable(test_glm test_glm.cpp)


add_executable(test_view_control test_view_control.cpp)
target_link_libraries(test_view_control 
    lib_visualize
)


add_executable(test_render_pointcloud test_render_pointcloud.cpp)
target_link_libraries(test_render_pointcloud
    lib_visualize
    lib_imgproc
)
