find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

## Use Eigen instead of GLM
# It seems glm is in /usr/include
# glm
# set(GLM_ROOT ${PROJECT_SOURCE_DIR}/glm-0.9.9.8)
# find_path(GLM_INCLUDE_DIRS DOC "Path to GLM include directory." NAMES glm/glm.hpp PATHS ${GLM_ROOT})
# message(STATUS "GLM_INCLUDE_DIRS:  ${GLM_INCLUDE_DIRS}")

add_library(lib_visualize
    gl_shader.cpp
    gl_render.cpp
    glad.c
    visualizer.cpp
)
target_link_libraries(lib_visualize
PRIVATE
    dl
)

