/**
 * image processor
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/26 16:51
 */
#include "image_proc.h"

ImageProcessor::ImageProcessor(int width, int height, double fx, double fy, double cx,
                               double cy) : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
{
    camera_intrinsic_inv_ = make_float4(1.0 / fx_, 1.0 / fy_, cx_, cy_);
    AllocateBuffer();
}

ImageProcessor::ImageProcessor(const CameraParams camera_params)
{
    width_ = camera_params.cols_;
    height_ = camera_params.rows_;
    fx_ = camera_params.fx_;
    fy_ = camera_params.fy_;
    cx_ = camera_params.cx_;
    cy_ = camera_params.cy_;
    camera_intrinsic_inv_ = make_float4(1.0 / fx_, 1.0 / fy_, cx_, cy_);
    AllocateBuffer();
}

void ImageProcessor::AllocateBuffer()
{
    depth_texture_surface_ = std::make_shared<CudaTextureSurface2D<ushort>>(height_, width_);
    vertex_texture_surface_ = std::make_shared<CudaTextureSurface2D<float4>>(height_, width_);
    depth_buffer_pagelock_ = std::make_shared<PagelockMemory>(sizeof(uint16_t) * height_ * width_);
    vertex_buffer_pagelock_ = std::make_shared<PagelockMemory>(sizeof(float4) * height_ * width_);
    // necessary to sync after cudaMallocHost?
    CudaSafeCall(cudaDeviceSynchronize());
    CudaSafeCall(cudaGetLastError());
}

void ImageProcessor::BuildVertexMap(const cv::Mat &depth_img)
{
    assert(depth_img.size().width == width_);
    assert(depth_img.size().height == height_);
    assert(depth_img.type() == 2); // "depth image type is expected to be CV_16UC1"
    depth_buffer_pagelock_->HostCopyFrom(static_cast<void *>(depth_img.data));
    depth_buffer_pagelock_->UploadToDevice(depth_texture_surface_->d_array_, stream_);
    ComputeVertex(make_ushort2(width_, height_),
                  camera_intrinsic_inv_,
                  depth_texture_surface_->texture_,
                  vertex_texture_surface_->surface_,
                  stream_);
}

// inside synchronize
void ImageProcessor::BuildVertexMap(const cv::Mat &depth_img, cv::Mat &vertex_map)
{
    BuildVertexMap(depth_img);
    vertex_buffer_pagelock_->DownloadFromDevice(vertex_texture_surface_->d_array_, stream_);
    stream_.Synchronize();
    CudaSafeCall(cudaGetLastError());
    vertex_buffer_pagelock_->HostCopyTo(vertex_map.data);
}