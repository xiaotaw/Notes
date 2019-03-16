### 0. 创建nvidia-docker
0.0 本机环境：ubuntu 18.04，CUDA 9.0，cuDNN 7.3，已安装docker(18.09.3)和nvidia-docker(2.0.3)  
0.1 拉取基础image  
`nvidia-docker pull  nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04`  
0.2 从image创建并运行container  
`nvidia-docker run -it -v /home/xt/Documents/data:/data nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 /bin/bash`  
0.3 测试  
`nvidia-smi`显示GPU信息    
`cat /usr/local/cuda/version.txt`显示CUDA信息  
CUDA Version 9.0.176  


## 在一个ubuntu16.04系统中(nvidia-docker)源码编译tensorflow1.8  


### 1. 安装编译环境(gcc-4.8)，及其他工具(vim git)  
`apt-get update`  
`apt-get install -y vim`  
`apt-get install -y git`  
`apt-get install -y gcc-4.8`

  ubuntu16.04默认gcc版本5.4，安装4.8版本后，重定向软链接  
`cd /usr/bin`  
`rm gcc gcc-ar gcc-nm gcc-ranlib`  
`ln -s gcc-4.8 gcc`  
`ln -s gcc-ar-4.8 gcc-ar`  
`ln -s gcc-nm-4.8 gcc-nm`  
`ln -s gcc-ranlib-4.8 gcc-ranlib`  

`cd && gcc -v` 检查版本

### 2. 按照官网的指导，安装python和pip (https://tensorflow.google.cn/install/source#build_the_package)  
`apt install -y python-dev python-pip`  
`pip install -U --user pip six numpy wheel setuptools mock`  

升级pip后，pip报错，参照https://blog.csdn.net/zong596568821xp/article/details/80410416修复  
`vim /usr/bin/pip`  
将原文  
```
from pip import main
if __name__ == '__main__':
    sys.exit(main())
```    
修改为  
```   
from pip import __main__ 
if __name__ == '__main__': 
    sys.exit(__main__._main())
```   
安装keras相关依赖包  
`pip install -U --user keras_applications==1.0.6 --no-deps`  
`pip install -U --user keras_preprocessing==1.0.5 --no-deps`  

### 3. 安装bazel，按照 https://docs.bazel.build/versions/master/install-ubuntu.html 中的第一种安装方式进行安装
3.1 安装依赖  
`apt-get install pkg-config zip g++ zlib1g-dev unzip python`  
3.2 下载bazel-0.10.0相应文件  
从Tags中选择0.10.0，下载bazel-0.10.0-installer-linux-x86_64.sh文件  
3.3 安装bazel  
`chmod +x bazel-0.10.0-installer-linux-x86_64.sh`  
`./bazel-0.10.0-installer-linux-x86_64.sh --user`  

因为是docker root用户，添加bazel所在路径到环境变量  
`export PATH=$PATH:/root/bin`

### 4. 添加GPU相关支持，本机已有nvidia驱动，Docker中已有CUDA，只需安装cuDNN
4.1 从nvidia官网下载安装cuDNN 7.3，并将解压得到的头文件cudnn.h、库文件libcudnn.so和libcudnn_static.a，分别复制到/usr/local/cuda中对应的文件夹下  
4.2 检查  
`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`  
#define CUDNN_MAJOR 7  
#define CUDNN_MINOR 3  
#define CUDNN_PATCHLEVEL 1  

### 5. 下载tensorflow源码，选择tensorflow版本
`git clone https://github.com/tensorflow/tensorflow.git`  
`cd tensorflow`  
`git checkout r1.8`

### 6. configure
`./configure`
jemalloc默认Y  
Google Cloud Platform \ Hadoop \ Amazon S3 \ Apache Kafka Platform选择N  
CUDA选择Y，CUDA version默认9.0，位置默认，cuDNN version输入7.3，位置默认  
compute capabilities输入6.1  
-march=native默认，表示对探测本机CPU架构并做相应优化  
其他不懂全选择默认

### 7. 使用bazel编译生成pip安装包
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
