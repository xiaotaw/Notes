/**
 * image processor
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/25 04:15
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <vector_functions.hpp> // for make_ushort2

#include "cuda_texture_surface.h"
#include "compute_vertex.h"
#include "dataset/camera_params.hpp"

class ImageProcessor
{
public:
    int width_, height_;
    double fx_, fy_, cx_, cy_;
    CudaTextureSurface2D<ushort>::Ptr depth_texture_surface_;
    CudaTextureSurface2D<float4>::Ptr vertex_texture_surface_;

    PagelockMemory::Ptr depth_buffer_pagelock_;
    PagelockMemory::Ptr vertex_buffer_pagelock_;

    CudaStream stream_;
    float4 camera_intrinsic_inv_;

    // ctor
    ImageProcessor(int width, int height, double fx, double fy, double cx, double cy);

    ImageProcessor(const CameraParams camera_params);

    void AllocateBuffer();

    void BuildVertexMap(const cv::Mat &depth_img);

    // inside synchronize
    void BuildVertexMap(const cv::Mat &depth_img, cv::Mat &vertex_map);
};