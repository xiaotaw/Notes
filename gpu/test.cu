#include "cuda_runtime.h"
#include <iostream>


__global__ void test(){
    //for(int i = 0; i < 10)

    //printf("a: %f", a);
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float a = float(index) / 10.0;
    int b = __float2int_rn(a);
    if (index <= 10)
        printf("%f: %d\n", a, b);
    if (index == 0){
        int value_i = 0x7fffffff;
        float value = *((float*)&(value_i));
        float value2  =  value * 2.0;
        float value3 = value + 12345.0;
        float value4 = 1.0 / value;
        printf("quiet_nan: %f, quiet_nan * 2: %f, quiet_nan + float: %f, 1.0 / quiet_nan: %f\n", 
            value, value2, value3, value4
        );
        float v = 123.0;
        if (v > value){
            printf("float %f is greater than float %f\n", v, value);
        }else{
            printf("float %f is not greater than float %f\n", v, value);
        }

        if (v < value){
            printf("float %f is less than float %f\n", v, value);
        }else{
            printf("float %f is not less than float %f\n", v, value);
        }
        if (value  == *((float*)&(value_i))){
            printf("nan == nan\n");
        }else{
            printf("nan != nan\n");
        }
        if(isnan(value)){
            printf("isnan(value): true\n");
        }
        //float
      

        ushort value5 = 0xFFFFFFFF;
        if(value5 == 0xFFFFFFFF){
            printf("ushort value5 == 0xFFFFFFFF\n");
        }
        ushort value6 = 0xFFFF;
        if(value6 == 0xFFFF){
            printf("ushort value6 == 0xFFFF\n");
        }
        float value7 = __expf(-value);
        printf("exp(-nan): %\nf", value7);
    }
}


// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int testAdd()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);
    
    test <<< gridSize, blockSize >>>();

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyHostToDevice);

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);

    return 0;
}


int main(){
    //test <<<dim3(2), dim3(2)>>>();
    testAdd();
    return 0;
}

