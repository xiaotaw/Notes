## 从源码编译tensorflow=1.14的pip安装包(2019/09/09)


* 历史版本：[从源码编译tensorflow=1.8的pip安装包(2019/03/16)](https://github.com/xiaotaw/Notes/blob/master/compile_tf18_from_source/20190316.md)

* 官方测试  

| Version	| Python version | Compiler	| Build tools	| cuDNN	| CUDA |
|-|-|-|-|-|-|
|tensorflow_gpu-1.14.0	| 2.7, 3.3-3.7 |	GCC 4.8	| Bazel 0.24.1 |	7.4 |	10.0 |
|tensorflow_gpu-1.8.0	| 2.7, 3.3-3.6	| GCC 4.8	| Bazel 0.10.0	| 7 |	9 |
* 个人尝试  

| Version	| Python version | Compiler	| Build tools	| cuDNN	| CUDA |
|-|-|-|-|-|-|
|tensorflow_gpu-1.14.0	| 2.7 |	GCC 5.4	| Bazel 0.26.1 |	7.4 |	10.1 |
|tensorflow_gpu-1.8.0	| 2.7	| GCC 4.8	| Bazel 0.10.0	| 7 |	9 |

* 百度云
Cuntomized Tensorflow python installation packages  
CPU optimization for 'Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz'  
For more info: https://github.com/xiaotaw/Notes/tree/master/compile_tf18_from_source  

tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl 链接: https://pan.baidu.com/s/1Lhh4z8rQ3OaP9aVc2i8nZw 提取码: hpyd 
tensorflow-1.14.1-cp27-cp27mu-linux_x86_64.whl 链接: https://pan.baidu.com/s/1Quyh7vXny4ahWSFGmkmsvA 提取码: yxcn   

## 目录
* [简介](#简介)
* [编译](#编译)
  * [0_创建容器](#0_创建容器)
  * [1_安装工具vim，git，gcc，g++](#1_安装工具vim，git，gcc，g++)
  * [2_安装python和tensorflow依赖库](#2_安装python和tensorflow依赖库)
  * [3_安装bazel](#3_安装bazel)
  * [4_检查GPU依赖](#4_检查GPU依赖)
  * [5_下载tensorflow源码，选择tensorflow版本](#5_下载tensorflow源码，选择tensorflow版本)
  * [6_configure](#6_configure)
  * [7_使用bazel编译生成pip安装包](#7_使用bazel编译生成pip安装包)
  * [8_安装测试](#8_安装测试)
* [参考资料](#参考资料)

## 简介
初衷：官方tensorflow的pip安装包，为了兼容不同CPU，没有启用一些加速运算的指令集。在运行tensorflow时，
会看到类似Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA的提示，
因此，自行编译启用这些指令，加速运算（有时间的话，会做对比测试）

## 编译
### 0_创建容器
*镜像可以从docker-hub获取*
```bash
# 获取镜像
docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
# 运行容器，-v指定文件映射，将本地/home/xt/Documents/data/映射到容器内的/data/。
nvidia-docker run -v /home/xt/Documents/data/:/data/ -it nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 /bin/bash
```
*以下基本都在容器内操作*

### 1_安装工具vim，git，gcc，g++
容器外，可提前准备[ubuntu16.04_sources.list](https://github.com/xiaotaw/Notes/tree/master/ubuntu/ubuntu16.04_sources.list)
```bash
# 用清华tuna镜像，替换官方apt源
cd /etc/apt/ && mv sources.list sources.list_bak && cp /data/ubuntu16.04_sources.list sources.list && cd
apt-get update
# 安装工具
apt-get install -y vim wget git gcc g++ 
```

### 2_安装python和tensorflow依赖库

```bash
apt install -y python-dev python-pip
pip install -U pip 
```
升级pip后，pip报错，参照https://blog.csdn.net/zong596568821xp/article/details/80410416 进行修复，使用vim打开/usr/bin/pip
```vim 
from pip import main
if __name__ == '__main__':
    sys.exit(main())
```
修改为
```vim
from pip import __main__ 
if __name__ == '__main__': 
    sys.exit(__main__._main())
```
继续安装tensorflow的依赖库
```bash
pip install -U six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U keras_applications==1.0.6 --no-deps
pip install -U keras_preprocessing==1.0.5 --no-deps
```
  
### 3_安装bazel
* 按照 https://docs.bazel.build/versions/master/install-ubuntu.html 中的第一种安装方式进行安装
```bash 
# 安装依赖
apt-get install -y pkg-config zip g++ zlib1g-dev unzip python 

# 下载bazel，按照https://tensorflow.google.cn/install/source#build_the_package的指示，
# 打开https://github.com/tensorflow/tensorflow/blob/master/configure.py，查看
# _TF_MIN_BAZEL_VERSION和_TF_MAX_BAZEL_VERSION的值，bazel版本需结余两者之间。这里选择0.26.1
wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh

# 安装bazel   
bash bazel-0.26.1-installer-linux-x86_64.sh 
```

### 4_检查GPU依赖
* 本机已有nvidia驱动，docker中有cuda和cudnn，因此不需安装。
```bash
# 检查nvidia驱动，得到 "NVIDIA-SMI 430.34       Driver Version: 430.34       CUDA Version: 10.1"
nvidia-smi
# 检查cuda版本，得到 "CUDA Version 10.1.243"
cat /usr/local/cuda/version.txt
# 检查cudnn版本，得到 "#define CUDNN_MAJOR 7    #define CUDNN_MINOR 6   #define CUDNN_PATCHLEVEL 3"
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

### 5_下载tensorflow源码，选择tensorflow版本
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.14
```

### 6_configure
`./configure`  
jemalloc默认Y  
Google Cloud Platform \ Hadoop \ Amazon S3 \ Apache Kafka Platform选择N  
CUDA选择Y，CUDA version默认9.0，位置默认，cuDNN version输入7.3，位置默认  
compute capabilities输入6.1  
-march=native默认，表示对探测本机CPU架构并做相应优化  
其他不懂全选择默认

```bash
./configure
XLA JIT
# 即时编译，还在实验阶段，并且在python代码中需要显式指明，N

OpenCL SYCL
# 需要配置额外的SYCL支持，N

ROCm
# amd gpu，咱不用，N

CUDA
# 用，Y

TensorRT
# 暂时不用吧，默认N

#CUDA SDK，cudnn版本自动找到。cuda10.1，cudnn7

CUDA compute capabilities 
# 6.1 (根据自己的显卡选择：https://developer.nvidia.com/cuda-gpus)

use clang as CUDA compiler
# No

MPI
# 不用多机并行运算，N

optimization flags
# 默认-march=native根据本机cpu自行优化

Android builds
# N

```

### 7_使用bazel编译生成pip安装包

```bash
# -config=opt: 针对cpu优化
# -config=cuda: 编译nvidia-gpu支持
# --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"，由于gcc版本是5，兼容旧版ABI，确保针对官方TensorFlow pip软件包编译的自定义操作继续支持使用GCC 5编译的软件包。
# --verbose_failures 好像开了也没啥用
# 编译时间有点长，12线程并行，大约90分钟。 
bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package --verbose_failures

# 报错 "No module named enum"，说是Python2的问题，毕竟python2快被抛弃了。
pip install --upgrade pip enum34

# 更新后，继续编译即可。

# 生成安装包 tensorflow-1.14.1-cp27-cp27mu-linux_x86_64.whl
./bazel-bin/tensorflow/tools/pip_package/build_pip_package  ../

# 清除
bazel clean 
```


### 8_安装测试
* 安装
```bash
# 安装tensorflow-gpu=1.14，环境需要cuda=10，cudnn=7，cpu是根据个人本机优化的。
docker pull nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04
nvidia-docker run -v /path/to/tensorflow-1.14.1-cp27-cp27mu-linux_x86_64.whl/:/data/ -it docker pull nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04 bash

# 安装python (如上文所述，可先更新apt源为国内镜像)，升级pip(如上文所述，修复pip的bug)
apt-get install python-dev python-pip
pip install -U pip

# 修复pip后，更新setuptools，tensorbord和markdown需要
pip install -U setuptools

# 安装tensorflow-gpu=1.14
cd /data/ && pip install tensorflow-1.14.1-cp27-cp27mu-linux_x86_64.whl

# 检查python版本，以及CPU指令集的优化
python -c "import tensorflow as tf; print(tf.__version__); print('\n\n'); sess=tf.Session()"
# 打印出1.14.1；以及一大段session的启动log，其中没发现类似于I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA之类的。
```

* 测试
(todo：用官方模型，跑对比试验，与官方的1.14版本对比运行速度)

## 参考资料
* tensorflow官网参考文档：https://tensorflow.google.cn/install/source#build_the_package
* pip的bug修复：https://blog.csdn.net/zong596568821xp/article/details/80410416
