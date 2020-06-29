add_library(lib_static_image_properties 
    static_image_properties.cpp
)
target_link_libraries(lib_static_image_properties
    PUBLIC
        k4a::k4a 
        ${OpenCV_LIBS}
)

add_library(lib_time_val time_val.cpp)

cuda_add_library(lib_cuda_common 
    cuda_texture_surface.cpp
    compute_vertex.cu
)
