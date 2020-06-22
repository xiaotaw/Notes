add_library(lib_static_image_properties 
    StaticImageProperties.cpp
)
target_link_libraries(lib_static_image_properties
    PUBLIC
        k4a::k4a 
        ${OpenCV_LIBS}
)

add_library(lib_time_val time_val.cpp)