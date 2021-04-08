add_library(lib_dataset 
    dataset.cpp
    dataset_base.cpp
    dataset_nyu_v2.cpp
    dataset_azure_kinect.cpp
)

target_link_libraries(lib_dataset
    PUBLIC
        ${OpenCV_LIBS} 
        ${Boost_SYSTEM_LIBRARY} 
        ${Boost_FILESYSTEM_LIBRARY}
        pthread
)