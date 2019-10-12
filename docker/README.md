## 目录
* [下载和上传镜像](#下载和上传镜像)
  * [官网](#官网)
  * [阿里云（登录使用）](#阿里云（登录使用）)
    * [镜像加速器](#镜像加速器)
    * [用户公开镜像](#用户公开镜像)
    * [将镜像上传至阿里云本地镜像仓库](#将镜像上传至阿里云本地镜像仓库)
* [创建镜像](#创建镜像)
* [保存镜像](#保存镜像)
* [使用容器](#使用容器)
  * [端口映射](#端口映射)
    * [指定端口映射](#指定端口映射)
    * [容器使用本地端口](#容器使用本地端口)
  * [docker图形界面](#docker图形界面)
  * [容器时间与宿主机同步](#容器时间与宿主机同步)
  * [异步进入容器](#异步进入容器)
* [个人容器镜像，存放于阿里云容器镜像服务器](#个人容器镜像，存放于阿里云容器镜像服务器)

## 下载和上传镜像
### 官网
* 浏览器登录[dockerhub](https://hub.docker.com/)，搜索CUDA，进入nvidia/CUDA
* 找到对应的docker pull command `docker pull nvidia/cuda`，直接使用将获取tag为latest的镜像
* 在overview页面，可以寻找最新镜像对应的tag，例如：10.1-cudnn7-runtime-ubuntu16.04
* 使用命令获取镜像：`docker pull nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 使用命令查看镜像：`docker images`    

REPOSITORY | TAG | IMAGE ID | CREATED | SIZE
-|-|-|-|-
nvidia/cuda | 10.1-cudnn7-runtime-ubuntu16.04 | b400c26dd64a | 7 days ago | 1.64GB

### 阿里云（登录使用）
#### 镜像加速器 
* 详情参考：https://cr.console.aliyun.com/cn-beijing/instances/mirrors
#### 用户公开镜像
* 详情参考：https://cr.console.aliyun.com/cn-beijing/instances/images
* All the images mentioned in this page are available in aliyun: [**Image List**](https://github.com/xiaotaw/Notes/blob/master/docker/images.md).

#### 将镜像上传至阿里云本地镜像仓库
* 登录[阿里云镜像服务管理控制台](https://cr.console.aliyun.com)，进入容器镜像服务
* 默认实例->访问凭证，设置固定密码
* 默认实例->命名空间，根据指引创建命名空间 xt-cuda
* 默认实例->镜像仓库，网页顶部左端，选择华北2(北京)，然后根据指引，在命名空间下，创建镜像仓库 cuda，不关联其他账号，使用本地镜像仓库，并设置为公开。
* 创建好的仓库->操作->管理中，参考操作指南，上传镜像。
* 登录Docker Registry: `docker login --username=xxxxxxxxxxx registry.cn-beijing.aliyuncs.com`，输入固定密码登录
* 将本地镜像标tag：`docker tag b400c26dd64a registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 上传至阿里云：`docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`




## 创建镜像
### 2.1 基于cuda:10.1-cudnn7-runtime-ubuntu16.04，创建pytorch1.1.0镜像
* 在容器外，下载anaconda3的安装包：`cd /home/xt/Documents/data/ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
* 创建并运行一个容器：`nvidia-docker run -v /home/xt/Documents/data/:/data/ -it nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04`，容器中有一个自带CUDA10和cudnn7的干净ubuntu16.04系统，没有其他多余的东西。 `-v`选项，将本地`/home/xt/Documents/data/`映射到容器内的`/data/`，刚下载的Miniconda3安装包就在容器内`/data/`目录下。
* 安装Miniconda3: `cd /data/ && bash Miniconda3-2019.07-Linux-x86_64.sh`
* 使用conda创建pytorch环境`conda create -n py3_7pytorch1_1_0 python=3.7`，并进入环境`source activate py3_7pytorch1_1_0`
* 按照[pytorch官网安装教程](https://pytorch.org/)，选择Stable(1.1), linux, conda, Python3.7, CUDA9.0，得到安装命令，复制并运行`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
* 检查版本`python3 -c "import torch; print(torch.__version__)"`得到1.1.0，`exit`退出容器
* `docker ps -a`获取容器信息，得到容器ID为`4f4cb2b6e9fa`
* 保存容器修改，存为为新镜像`nvidia-docker commit -a xt -m "create pytorch1.1.0 based on nvidia/cuda" 4f4cb2b6e9fa nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0`，`docker images`获取新镜像ID为`2b197ded9713`
* 将本地镜像标tag：`docker tag 2b197ded9713 registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0`
* 上传至阿里云：`docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0`。
* 新镜像安装了Miniconda3，python3，pytorch1.1，内容大了不少，有5.5GB，呵呵，上传有点慢（开始使用anaconda3,7.5G，呵呵呵）。
* **更新：安装fairseq** 安装fairseq之前，注意需`source activate py3_7pytorch1_1_0`；另外依赖于gcc，`apt-get update && apt-get install build-essential`；最后`pip install fairseq`；上传至阿里云：`docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0_gcc_fairseq`，由于很多Layer已经存在，只需上传200M，比较快。
* **更新：安装vim** `apt-get install vim`；上传至阿里云: `docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0_gcc_fairseq_vim`

## 保存镜像
docker save -o xxx.tar IMAGE

## 使用容器

### 端口映射
#### 指定端口映射
将容器的80端口，映射到本地8080端口: `docker run -p 8080:80`
#### 容器使用本地端口
网络连接方式默认为 brige，修改为host后，容器localhost指向容器宿主机，不需要使用-p或者-P指定端口。容器内开启的端口，宿主机也被占用。
`docker run --network="host"`

### docker图形界面
```bash
# 开放x11本地访问权限
xhost +local:root

# 共享x11端口
docker run -ti --rm  -e DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix  ubuntu bash

# 简单测试
apt-get install xarclock

xarclock
# 运行后出现时钟
```
https://blog.csdn.net/ericcchen/article/details/79253416


### 容器时间与宿主机同步
1. 运行容器时，添加 -v /etc/localtime:/etc/localtime:ro


### 异步进入容器
```bash
# 
docker exec -it container_id /bin/bash

docker exec -it container_id /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```

## 个人容器镜像，存放于阿里云容器镜像服务器

地区 |主题 | 仓库 | tag | 备注 | Digest
-|-|-|-|-|-
华东1（杭州）| TensorRT | registry.cn-hangzhou.aliyuncs.com/tensorrt/tensorrt | 19.02-py2_tf1.14 | tensorrt+tensorflow1.14 | adf5ed9b87e7e8bf87375b842558337ab4bcb957f8c57564ea5bf01641f42eff 
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-runtime-ubuntu16.04 | 带CUDA的ubt系统镜像 | e6b365c666f3c161202f62b9a054a78bd030e94eb31adf65c458f937523f548e 
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0 | 带CUDA的ubt系统镜像，添加miniconda3，torch1.1.0 | e8a6faa2ba5f036979951902ef0e3d39582fd2f7e9b8a899170daa87a5a8706d
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0_gcc_fairseq | 带CUDA的ubt系统镜像，添加miniconda3，torch1.1.0，以及gcc，fairseq | efa99afd282cf7cd67ca0171e423d09acc750998b2f0ce5bcfe484ee68db3b5e
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-runtime-ubuntu16.04-miniconda3-py37-pytorch1_1_0_gcc_fairseq_vim | 带CUDA的ubt系统镜像，添加miniconda3，torch1.1.0，以及gcc，fairseq,vim | 40242d03859d011585893cc90df2df0b1fc9c8dd4f54fb1c42ab6815240527eb
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-devel-ubuntu16.04 | \ | \
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-devel-ubuntu16.04-building_tf_whl | 配置tensorflow的pip安装包编译环境 | \
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-devel-ubuntu16.04-building_tf_whl_r1.14 | 编译了r1.14，并执行bazel clean，大小缩减了许多 | \
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 10.1-cudnn7-runtime-ubuntu16.04-tf_gpu1.14_opted | 安装tensorflow-gpu=1.14，cpu指令集SSE4.1 SSE4.2 AVX AVX2 FMA | \
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/cuda | 9.0-cudnn7-devel-ubuntu16.04_tf_py3_whl_building_env | 配置了cuda9.0，cudnn7的tensorflow_py3.whl编译环境 | \
华北2（北京）| CUDA | registry.cn-beijing.aliyuncs.com/xt-cuda/gibson | 0.3.1_tf_cpu_optmized | 在xf1280/gibson:0.3.1的基础上，更新tensorflow版本至1.8，同时有针对我个人电脑cpu优化（虽然测试没看出明显加速效果） | \

地区 |主题 | 仓库 | tag | 备注 | Digest
-|-|-|-|-|-
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04 | basic | 93b34b7632eed4e9909cf7a140e162cdf1bbd984aef49b24cb1f7e0d6e2d67d0
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-miniconda2 | \ | \
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-miniconda2-flask | \ | \
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-pipenv | \ | \
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-pipenv-fun | \ | \
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-neo4j | \ | \
华北2（北京）| web | registry.cn-beijing.aliyuncs.com/xt-web/web | ubuntu16.04-neo4j-vim | \ | \
