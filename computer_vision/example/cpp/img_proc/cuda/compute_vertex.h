#pragma once
#include "img_proc/cam_intr.h"
#include "img_proc/cuda/containers/device_array.hpp"
#include "img_proc/cuda/cuda_texture_surface.h"
#include <cuda_runtime_api.h>

void ComputeVertex(CudaTextureSurface2D<ushort>::Ptr depth,
                   CudaTextureSurface2D<float4>::Ptr vertex,
                   const CamIntrInv &cam_intr_inv, cudaStream_t stream = 0);

void ComputeVertex(const DeviceArray2D<float> depth,
                   DeviceArray2D<float> vertex, const CamIntrInv &cam_intr_inv,
                   cudaStream_t stream = 0);