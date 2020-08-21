
add_executable(test_glfw_glad test_glfw_glad.cpp)
target_link_libraries(test_glfw_glad 
    glfw 
    lib_visualize
)


add_executable(test_triangle test_triangle.cpp)
target_link_libraries(test_triangle 
    glfw 
    lib_visualize
)
