/**
 * cuda texture surface 
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/29 22:34
 */
#include <memory.h>
#include "cuda_runtime_api.h"
#include "common/cuda_texture_surface.h"

void CudaTextureSurface2D::CreateTextureSurface()
{
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

void CudaTextureSurface2D::DestoryTextureSurface(){
    CudaSafeCall(cudaDestroyTextureObject(texture_));
    CudaSafeCall(cudaDestroySurfaceObject(surface_));
    CudaSafeCall(cudaFreeArray(d_array_));

}