## 目录
* [从源码编译tensorflow=1.14的pip安装包(2019/09/09)](#从源码编译tensorflow的pip安装包)
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
* [源码编译tensorflow=1.8动态库libtensorflow_cc.so](#源码编译tensorflow动态库)


## 从源码编译tensorflow的pip安装包

初衷：官方tensorflow的pip安装包，为了兼容不同CPU，没有启用一些加速运算的指令集。在运行tensorflow时，
会看到类似Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA的提示，
因此，自行编译启用这些指令，加速运算（有时间的话，会做对比测试）


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
|tensorflow_gpu-1.8.0	| 3.5	| GCC 4.8	| Bazel 0.10.0	| 7 |	9 |

* 百度云
Cuntomized Tensorflow python installation packages  
CPU optimization for 'Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz'  
For more info: https://github.com/xiaotaw/Notes/tree/master/compile_tf18_from_source  

tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl 链接: https://pan.baidu.com/s/1Lhh4z8rQ3OaP9aVc2i8nZw 提取码: hpyd 
tensorflow-1.14.1-cp27-cp27mu-linux_x86_64.whl 链接: https://pan.baidu.com/s/1Quyh7vXny4ahWSFGmkmsvA 提取码: yxcn   

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
# for python2
apt install -y python-dev python-pip
pip install -U pip 

# for python3
apt install -y python3-dev python3-pip
#pip3 install -U pip3 (有问题)
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
# for python2
# 若是国内可使用清华源 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U keras_applications==1.0.6 --no-deps
pip install -U keras_preprocessing==1.0.5 --no-deps


# for python3
# 若是国内可使用清华源 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -U six numpy wheel setuptools mock 'future>=0.17.1'
pip3 install -U keras_applications==1.0.6 --no-deps
pip3 install -U keras_preprocessing==1.0.5 --no-deps
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
# 检查cuda版本，得到类似 "CUDA Version 10.1.243" 结果。
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

### 参考资料
* tensorflow官网参考文档：https://tensorflow.google.cn/install/source#build_the_package
* pip的bug修复：https://blog.csdn.net/zong596568821xp/article/details/80410416


## 源码编译tensorflow动态库

* 在centos6中，使用bazel编译c版本的tensorflow
* 因为生产环境中，GLIBC版本较低（大约是contos6.5，GLIBC版本为2.12），因此需自主编译tensorflow的动态库。

### docker创建系统环境
* contos6.5是非长期维护版本，这里使用centos6代替，各软件的支持较多，比如可以直接从nvidia/cuda获取contos6的镜像，但是没有contos6.5的。
* github上很多关于cuda10.1不兼容的issues，用cuda10.0的版本更合适。
```bash
# centos6 + cuda10 + cudnn7
docker pull nvidia/cuda:10.0-cudnn7-devel-centos6 

# -v /etc/localtime:/etc/localtime:ro 是为了容器内和宿主机时间同步，可以省略
# -v /home/xt/Documents/data/:/data/ 是为了方便数据保存备份，有些软件下载较慢，可以备份至宿主机，今后方便重用。
nvidia-docker run -it -v /home/xt/Documents/data/:/data/ -v /etc/localtime:/etc/localtime:ro nvidia/cuda:10.0-cudnn7-devel-centos6

# 以下均在容器内运行
```

* 查找并定位libc和libstdc++库

```bash
# 不更新一下，还找不到libc.so
yum update

ldconfig -p | grep libc.so
# 得到libc库的位置：/lib64/libc.so.6

ldconfig -p | grep libstdc++.so
# 得到libc库的位置：/usr/lib64/libstdc++.so.6
```

* 检查GLIBC和GLIBCXX的版本
```bash
strings /lib64/libc.so.6 | grep GLIBC
# 得知GLIBC版本最高支持2.12，比ubuntu18.04低（GLIBC_2.27）

strings /usr/lib64/libstdc++.so.6 | grep GLIBC
# 得知GLIBC版本最高支持3.4.13，比ubuntu18.04低（GLIBCXX_3.4.25）
```

* 用这种低版本的GLIBC和GLIBCXX编译出来的tensorflow动态库，兼容性应该很好。

### 编译安装bazel
* 不论是python版还是c/c++版，bazel是编译tensorflow的必备工具。bazel不同版本之间，兼容性有点问题。最好根据tensorflow的版本，选择合适的bazel的版本。

* 这里准备编译tensorflow1.8，对应使用的bazel版本为0.10.0，gcc版本为4.8。（参考编译python版tensorflow的案例，https://tensorflow.google.cn/install/source#tested_build_configurations）

* centos6使用`yum install bazel`6默认安装的bazel版本可能过高，并且依赖高版本的GLIBC，不可行。
* 下载bazel的0.10.0版本的linux安装包，安装也失败。
* 最后采用源码编译bazel的方式。参考https://docs.bazel.build/versions/master/install-compile-source.html#bootstrap-bazel。
* 这里需注意的是，不能从github上直接下载源码，得从release中选择[0.10.0版本](https://github.com/bazelbuild/bazel/releases/tag/0.10.0)，选择下载[bazel-0.10.0-dist.zip](https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip)文件，解压后编译。

* 首先是gcc
```bash
# 查看centos6默认gcc版本为4.4.7
gcc --version

# 安装一个额外的gcc4.8
# 参考https://blog.csdn.net/weixin_34384681/article/details/91921751
# 下载安装很慢，得很长一段时间。
yum install wget
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O \
     /etc/yum.repos.d/devtools-2.repo
yum install devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-gfortran devtoolset-2-gcc-c++

# 切换至gcc4.8，并查看gcc版本，得知gcc版本为4.8.2
scl enable devtoolset-2 bash
gcc --version

# 退出，至gcc4.4.7的版本
exit
```

* 其次bazel依赖于java
```bash
# 下载jdk 8的linux版，[oracle官网](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)下载不方便，这里提供一个百度网盘链接。
# 链接: https://pan.baidu.com/s/1NI7k_QYCXa8ZN9oQv7PA2w 提取码: 6zg9 复制这段内容后打开百度网盘手机App，操作更方便哦
# 从百度网盘中下载 jdk-8u172-linux-x64.tar.gz（手机下载速度较快）

tar jdk-8u172-linux-x64.tar.gz

# 简单配置java环境变量
export JAVA_HOME=`pwd`/jdk1.8.0_172
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# 也可以讲jdk放到/usr/local下,
mv jdk1.8.0_172 /usr/local
export JAVA_HOME=/usr/local/jdk1.8.0_172
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# 检查java
java -version
javac -version

# export只是临时添加环境变量；
# 更好的方式是，打开文件~/.bashrc，在末尾写入export语句，关闭保存，今后每次开启终端，都有java环境变量。
# source ~/.bashrc，使改动立即生效。
```


* 最后编译安装bazel
```bash
wget https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip
unzip bazel-0.10.0-dist.zip

cd bazel-0.10.0-dist

# 注意此时gcc需切换到4.8。
scl enable devtoolset-2 bash
bash compile.sh

# 编译得到output/bazel，bazel的路径添加至环境变量中，
export PATH=$PATH:`pwd`/output
which bazel

# 检查bazel版本为0.10.0
bazel version
```

### 准备tensorflow源码

* 准备1.8版本的tensorflow的源码
```bash
yum install git
# 从下载tensorflow源码
git clone https://github.com/tensorflow/tensorflow.git
# 切换到1.8版本
git checkout r1.8

# 在宿主机中已经提前下载好，通过docker的文件映射，放在容器/data路径下，可以省不少事。
```

### 安装tensorflow的依赖protobuf和eigen
* 安装依赖protobuf
* 参考文章：https://www.jianshu.com/p/d46596558640
```bash
# 安装automake和cmake
yum install autoconf automake libtool cmake

./tensorflow/contrib/makefile/download_dependencies.sh
# 下载不全没关系，protobuf和eigen下载了就行。

# protobuf
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure --prefix=/tmp/proto/
make -j8 && make install 


# eigen
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install
cd ../../../../../..

```

### 安装python等

```bash
# 
yum install centos-release-scl

yum install python27 python27-numpy python27-python-devel python27-python-wheel

### 编译
```

```
yum install git

yum install patch

# 报错: undefined reference to 'clock_gettime'
# 直接在bazel 命令中添加 --linkopt=-lrt 无效
# 参考：https://github.com/tensorflow/tensorflow/issues/15129
# 修改tensorflow/tensorflow.bzl，
def tf_cc_shared_object(
    name,
    srcs=[],
    deps=[],
    linkopts=[''],
    framework_so=tf_binary_additional_srcs(),
    **kwargs):
  native.cc_binary(
      name=name,
      srcs=srcs + framework_so,
      deps=deps,
      linkshared = 1,
      linkopts=linkopts + _rpath_linkopts(name) + select({
          clean_dep("//tensorflow:darwin"): [
              "-Wl,-install_name,@rpath/" + name.split("/")[-1],
          ],
          clean_dep("//tensorflow:windows"): [],
          "//conditions:default": [
              "-Wl,-soname," + name.split("/")[-1],
          ],
      }),
      **kwargs)
中的linkopts中添加'-lrt'，即：

def tf_cc_shared_object(
    name,
    srcs=[],
    deps=[],
    linkopts=['-lrt'],
    framework_so=tf_binary_additional_srcs(),
    **kwargs):
  native.cc_binary(
      name=name,
      srcs=srcs + framework_so,
      deps=deps,
      linkshared = 1,
      linkopts=linkopts + _rpath_linkopts(name) + select({
          clean_dep("//tensorflow:darwin"): [
              "-Wl,-install_name,@rpath/" + name.split("/")[-1],
          ],
          clean_dep("//tensorflow:windows"): [],
          "//conditions:default": [
              "-Wl,-soname," + name.split("/")[-1],
          ],
      }),
      **kwargs)

```

https://github.com/tensorflow/tensorflow/issues/15129
```















