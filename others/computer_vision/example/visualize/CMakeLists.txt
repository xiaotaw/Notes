find_package(glfw3 REQUIRED)


add_library(lib_visualize
    gl_shader.cpp
    gl_render.cpp
    glad.c
)
target_link_libraries(lib_visualize
PRIVATE
    dl
)

