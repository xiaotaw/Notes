/**
 * cuda texture surface
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/29 22:35
 */
#pragma once
#include <memory>
#include <channel_descriptor.h> // for cudaCreateChannelDesc<T>()
#include "common/cuda_snippets.hpp"

// (xt) TODO: move enum outside of template class, because it's independent of the template
// bilinear is not supported as filterMode only support cudaFilterModePoint, when data type is not float
// kInterArea is not supported yet.
enum InterpolationMode
{
    kNearest = 0, // cv::INTER_NEAREST
    //kBilinear = 1,   // cv::INTER_LINEAR not supported now
    kInterArea = 3 // cv::INTER_AREA
};

template <typename T>
class CudaTextureSurface2D
{
public:
    // just let them public
    cudaTextureObject_t texture_;
    cudaSurfaceObject_t surface_;
    cudaArray_t d_array_;
    unsigned rows_;
    unsigned cols_;

    // ctor
    CudaTextureSurface2D() = default;

    // ctor
    CudaTextureSurface2D(const unsigned rows, const unsigned cols) : rows_(rows), cols_(cols)
    {
        m_channel_desc = cudaCreateChannelDesc<T>();
        // allocate cuda array
        CudaSafeCall(cudaMallocArray(&d_array_, &m_channel_desc, cols_, rows_));
        // create resource desc
        memset(&m_resource_desc, 0, sizeof(m_resource_desc));
        m_resource_desc.resType = cudaResourceTypeArray;
        m_resource_desc.res.array.array = d_array_;
        // create surface
        CudaSafeCall(cudaCreateSurfaceObject(&surface_, &m_resource_desc));
        // create texture desc
        memset(&m_texture_desc, 0, sizeof(m_texture_desc));
        m_texture_desc.addressMode[0] = m_texture_desc.addressMode[1] = m_texture_desc.addressMode[2] = cudaAddressModeBorder;
        m_texture_desc.filterMode = cudaFilterModePoint;
        m_texture_desc.readMode = cudaReadModeElementType;
        m_texture_desc.normalizedCoords = 0;
        // create texture
        CudaSafeCall(cudaCreateTextureObject(&texture_, &m_resource_desc, &m_texture_desc, 0));
    }

    using Ptr = std::shared_ptr<CudaTextureSurface2D<T>>;

    // dtor
    ~CudaTextureSurface2D()
    {
        CudaSafeCall(cudaDestroyTextureObject(texture_));
        CudaSafeCall(cudaDestroySurfaceObject(surface_));
        CudaSafeCall(cudaFreeArray(d_array_));
    }

    DISABLE_COPY_ASSIGN_MOVE(CudaTextureSurface2D);

    //
    void ResizeDown(CudaTextureSurface2D &dst, cudaStream_t stream, InterpolationMode mode);
    void ResizeDownNearest(CudaTextureSurface2D &dst, cudaStream_t stream);
    void ResizeDownInterArea(CudaTextureSurface2D &dst, cudaStream_t stream);

private:
    cudaChannelFormatDesc m_channel_desc;
    cudaResourceDesc m_resource_desc;
    cudaTextureDesc m_texture_desc;
};
