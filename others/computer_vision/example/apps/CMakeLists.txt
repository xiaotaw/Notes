
ADD_EXECUTABLE(record_images record_images.cpp)
TARGET_LINK_LIBRARIES(record_images 
    lib_static_image_properties 
    lib_time_val
)