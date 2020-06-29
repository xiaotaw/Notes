#include "common/compute_vertex.h"

namespace device{
    /**
     * image_size = (image_cols, image_rows) = (image_width, image_height)
     * camera_intrinsic = (fx, fy, cx, cy)
     * camera_intrinsic_inv = (1/fx, 1/fy, cx, cy)
     * Note: vertex unit is millimeter(mm)
     */
    __global__ void ComputeVertexKernel(const ushort2 image_size, const float4 camera_intrinsic_inv, 
        cudaTextureObject_t depth_texture, cudaSurfaceObject_t vertex_surface)
    {
        const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
        const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= image_size.x || y >= image_size.y) return;

        // Note: Even if (vertex_z == 0); then vertex_x = vertex_y = 0.
        const float vertex_z = static_cast<float>(tex2D<ushort>(depth_texture, x, y));

        // V_x = (u - cx) * V_z / fx
        // V_y = (v - cy) * V_z / fy
        const float vertex_x = (x - camera_intrinsic_inv.z) * vertex_z * camera_intrinsic_inv.x;
        const float vertex_y = (y - camera_intrinsic_inv.w) * vertex_z * camera_intrinsic_inv.y;

        float4 vertex = make_float4(vertex_x, vertex_y, vertex_z, 1);
        surf2Dwrite(vertex, vertex_surface, x * sizeof(float4), y);
    }
} // namespace device








void ComputeVertex(const ushort2 image_size, const float4 camera_intrinsic_inv, 
    cudaTextureObject_t depth_texture, cudaSurfaceObject_t vertex_surface, cudaStream_t stream)
{
    dim3 blk(16, 16);
    dim3 grid(DivideUp(image_size.x, blk.x), DivideUp(image_size.y, blk.y));

    device::ComputeVertexKernel<<<grid, blk, 0, stream>>>(
        image_size,
        camera_intrinsic_inv,
        depth_texture,
        vertex_surface
    );
	CudaSafeCall(cudaStreamSynchronize(stream));
	CudaSafeCall(cudaGetLastError());
}