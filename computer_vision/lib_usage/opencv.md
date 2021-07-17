## 目录

- [目录](#目录)
- [ubuntu=18.04安装opencv](#ubuntu1804安装opencv)
  - [安装opencv4.2.0+CUDA](#安装opencv420cuda)
  - [安装opencv4.1.0](#安装opencv410)
  - [安装opencv3.1.0](#安装opencv310)
  - [安装opencv3.2.0](#安装opencv320)
- [参考资料](#参考资料)

## ubuntu=18.04安装opencv

*编译OpenCV时，会下载不少东西，且不同版本的源码可能需做一部分修改，因此若需编译安装某个版本，建议直接下载该版本release源码包，而不是通过git checkout切换。*


### 安装opencv4.2.0+CUDA
```bash
mkdir opencv-4.2.0
wget https://github.com/opencv/opencv/archive/refs/tags/4.2.0.tar.gz
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.2.0.tar.gz
tar zxvf opencv-4.2.0.tar.gz
tar zxvf opencv_contrib-4.2.0.tar.gz

cd opencv-4.2.0
mkdir build && cd build
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.2.0 \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules \
-D WITH_VTK=ON \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=6.1 \
-D CUDA_ARCH_PTX=6.1 \
-D ENABLE_FAST_MATH=1   \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PREF_TESTS=OFF \
..

make -j
sudo make install
```


### 安装opencv4.1.0

```bash
# /data为某一目录
cd /data 
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# checkout至某个版本，以4.1.0为例
cd opencv
git checkout 4.1.0
cd ..

# opencv_contrib版本与opencv保持一致
cd opencv_contrib
git checkout 4.1.0
cd ..

# 安装在/usr/local/opencv-4.1.0，这个路径在后续工程中需要使用
# 如同时使用opencv_contrib，则OPENCV_EXTRA_MODULES_PATH=<your path>/opencv_contrib/modules
# 如使用VTK，则需下载并编译VTK（https://vtk.org/download/），这里使用了VTK-8.2.0
# 然后设置VTK_DIR为VTK的cmake目录(该目录下包含VTKConfig.cmake)
# OPENCV_ENABLE_NONFREE=ON为了使用SIFT和SURF
# BUILD_EXAMPLES BUILD_TESTS BUILD_PREF_TESTS禁用
# It is useful also to unset BUILD_EXAMPLES, BUILD_TESTS, and BUILD_PERF_TESTS,
# as they all will be statically linked with OpenCV and can take a lot of memory.

cd opencv
mkdir build && cd build
cmake \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.1.0 \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D WITH_VTK=ON \
-D VTK_DIR=../../VTK-8.2.0/build \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PREF_TESTS=OFF \
..

make -j${nproc}
sudo make install
```

如果需要切换使用该版本，则在~/.bashrc中添加如下几行，并执行`source ~/.bashrc`之后`pkg-config --modversion opencv`查询验证。

```vim
#opencv-4.1.0
OPENCV_PATH=/usr/local/opencv-4.1.0
export PATH=${OPENCV_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${OPENCV_PATH}/lib:$LD_LIBRARY_PATH 
export PKG_CONFIG_PATH=${OPENCV_PATH}/lib/pkgconfig
```

在其他工程中使用该版本的opencv，在CMakeList.txt文件中添加

```vim
set(OpenCV_DIR "/usr/local/opencv-4.1.0")
find_package(OpenCV REQUIRED)
```

### 安装opencv3.1.0

*参照410的安装过程，稍有改动。*  

1. CUDA9以及以上的版本与opencv310不兼容，需修改cmake比较麻烦，这里先不使用cuda。修改方案可参考
[CSDN博客 OpenCV3.3+CUDA9.0+Cmake3.9 环境搭建](https://blog.csdn.net/u014613745/article/details/78310916)
2. gcc7将stdlib.h放入libstdc++以进行更好的优化，C Library的头文件stdlib.h使用 Include_next，而include_next对gcc系统头文件路径很敏感。推荐的修复方法是不要把include路径作为系统目录，而是使用标准方式包含include目录。参考来源
[CSDN博客 /usr/include/c++/7/cstdlib:75:15: fatal error: stdlib.h: No such file or directory](https://blog.csdn.net/u010003609/article/details/100086151)

```bash
cmake \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.1.0 \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D WITH_VTK=ON \
-D VTK_DIR=../../VTK-8.2.0/build \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PREF_TESTS=OFF \
-D WITH_CUDA=OFF \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
..
```

### 安装opencv3.2.0
- WITH_CUDA 失败  
  CMake Error: The following variables are used in this project, but they are set to NOTFOUND.  
  Please set them or make sure they are set and tested correctly in the CMake files:  
  CUDA_nppi_LIBRARY (ADVANCED)
- 报错: CMake Error at cmake/OpenCVCompilerOptions.cmake:21 (else):  
  解决：https://blog.csdn.net/weixin_41674487/article/details/88237764  

```bash
# without contrib
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.2.0 \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PREF_TESTS=OFF \
-D WITH_CUDA=OFF \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
..
```

```bash
# with contrib
# contrib中需下载一些诸如机器学习模型等文件，可使用proxy工具加速加载。
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.2.0 \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON \
-D BUILD_TESTS=ON \
-D BUILD_PREF_TESTS=ON \
-D WITH_CUDA=OFF \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
..
```


## 参考资料

1. [Ubuntu下多版本OpenCV共存和切换](https://blog.csdn.net/learning_tortosie/article/details/80594399?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
2. [ubuntu16.04 opencv(各个版本编译安装，来回切换)](https://blog.csdn.net/mhsszm/article/details/88558470)
3. [Azure Kinect - OpenCV KinectFusion Sample](https://github.com/microsoft/Azure-Kinect-Samples/tree/master/opencv-kinfu-samples)
