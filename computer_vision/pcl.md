

### 编译pcl 1.8.1 + CUDA 10.0
```bash
wget https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.8.1.tar.gz

tar pcl-1.8.1.tar.gz
cd pcl-pcl-1.8.1

mkdir build && cd build
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_GPU=ON \
-DCMAKE_INSTALL_PREFIX=../install \
-DBUILD_apps=ON \
-DBUILD_examples=ON \
-DCUDA_ARCH_BIN=6.1 \
-DCUDA_ARCH_PTX=6.1 \
-DCUDA_NVCC_FLAGS=-arch=sm_61 


mkdir build && cd build
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_GPU=ON \
-DCMAKE_INSTALL_PREFIX=../install_eigen3.3.9 \
-DBUILD_apps=ON \
-DBUILD_examples=ON \
-DCUDA_ARCH_BIN=6.1 \
-DCUDA_ARCH_PTX=6.1 \
-DCUDA_NVCC_FLAGS=-arch=sm_61

make -j
```

