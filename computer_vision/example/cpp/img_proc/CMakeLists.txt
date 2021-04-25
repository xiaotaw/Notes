cuda_add_library(lib_imgproc
  cuda/cuda_texture_surface.cu
  cuda/compute_vertex.cu
  cuda/compute_normal.cu
  cuda/bilateral_filter.cu
  cuda/containers/initialization.cpp
  cuda/containers/device_memory.cpp
  cuda/icp/icp.cu
  cuda/icp/icp.cpp
  image_proc.cpp
)

target_link_libraries(lib_imgproc
  k4a::k4a 
  ${OpenCV_LIBS}
  cuda
  lib_common
)