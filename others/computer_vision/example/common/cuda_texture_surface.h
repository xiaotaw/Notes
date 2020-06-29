/**
 * cuda texture surface
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/29 22:35
 */
#pragma once
#include <memory>
#include "common/cuda_snippets.hpp"

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
    CudaTextureSurface2D(const unsigned rows, const unsigned cols) : rows_(rows), cols_(cols) {}

    // call CreateChannelDesc before CreateTextureSurface
    virtual void CreateChannelDesc() = 0;
    // call CreateChannelDesc before CreateTextureSurface
    void CreateTextureSurface();
    void DestoryTextureSurface();

    cudaChannelFormatDesc m_channel_desc;
    cudaResourceDesc m_resource_desc;
    cudaTextureDesc m_texture_desc;
};

class CudaShortTextureSurface2D : public CudaTextureSurface2D
{
public:
    // ctor
    CudaShortTextureSurface2D(const unsigned rows, const unsigned cols) : CudaTextureSurface2D(rows, cols)
    {
        CreateChannelDesc();
        CreateTextureSurface();
    }

    using Ptr = std::shared_ptr<CudaShortTextureSurface2D>;

    inline void CreateChannelDesc() override
    {
        m_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    }

    // dtor
    ~CudaShortTextureSurface2D(){
        DestoryTextureSurface();
    }

};

class CudaFloat4TextureSurface2D : public CudaTextureSurface2D
{
public:
    // ctor
    CudaFloat4TextureSurface2D(const unsigned rows, const unsigned cols) : CudaTextureSurface2D(rows, cols)
    {
        CreateChannelDesc();
        CreateTextureSurface();
    }

    using Ptr = std::shared_ptr<CudaShortTextureSurface2D>;

    inline void CreateChannelDesc() override
    {
        m_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    }

    // dtor
    ~CudaFloat4TextureSurface2D(){
        DestoryTextureSurface();
    }

};