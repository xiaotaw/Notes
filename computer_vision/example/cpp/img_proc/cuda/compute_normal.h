/**
 * @file compute_normal.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-18
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include "img_proc/cuda/containers/device_array.hpp"
#include <cuda_runtime_api.h>

void ComputeNormal(const DeviceArray2D<float> vertex_map,
                   DeviceArray2D<float> normal_map, cudaStream_t stream = 0);