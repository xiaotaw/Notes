add_library(lib_common 
    time_val.cpp
    time_logger.cpp
    static_image_properties.cpp
)

target_link_libraries(lib_common
PUBLIC
    k4a::k4a
)