#pragma once
#include "common/cuda_texture_surface.h"

void ComputeVertex(const ushort2 image_size, const float4 camera_intrinsic_inv, cudaTextureObject_t depth_texture, cudaSurfaceObject_t vertex_surface, cudaStream_t stream = 0);