/**
 * @file bilater_filter.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-09 file create, add todo
 *       2021-04-11 complete todo, initial version
 * @copyright Copyright (c) 2021
 */
#pragma once

#include "img_proc/cuda/containers/device_array.hpp"
#include "img_proc/cuda/cuda_texture_surface.h"

/**
 * @brief bilateral filter on depth image
 *  !!! this function has not been tested yet !!!
 *
 * @param[in] depth_in
 * @param[out] depth_out
 * @param[in] sigma_d
 * @param[in] sigma_r
 */
void BilateralFilter(CudaTextureSurface2D<ushort> depth_in,
                     CudaTextureSurface2D<float> depth_out,
                     const float &sigma_d, const float &sigma_r,
                     cudaStream_t = 0);

void BilateralFilter(const DeviceArray2D<ushort> depth_in,
                     DeviceArray2D<float> depth_out, const float &sigma_d,
                     const float &sigma_r, cudaStream_t = 0);