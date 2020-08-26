add_library(lib_time 
    time_val.cpp
    time_logger.cpp
)

cuda_add_library(lib_imgproc
    cuda_texture_surface.cu
    compute_vertex.cu
    image_proc.cpp
    pyramid.hpp
    static_image_properties.cpp
)

target_link_libraries(lib_imgproc
    k4a::k4a 
    ${OpenCV_LIBS}
    cuda
)