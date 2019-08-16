# 1 获取/上传镜像
## 1.1 官网
* 浏览器登录[dockerhub](https://hub.docker.com/)，搜索CUDA，进入nvidia/CUDA
* 找到对应的docker pull command `docker pull nvidia/cuda`，直接使用将获取tag为latest的镜像
* 在overview页面，可以寻找最新镜像对应的tag，例如：10.1-cudnn7-runtime-ubuntu16.04
* 使用命令获取镜像：`docker pull nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 使用命令查看镜像：`docker images`    

REPOSITORY | TAG | IMAGE ID | CREATED | SIZE
-|-|-|-|-
nvidia/cuda | 10.1-cudnn7-runtime-ubuntu16.04 | b400c26dd64a | 7 days ago | 1.64GB

## 1.2 阿里云（登录使用）
#### 1.2.1 镜像加速器 
* 详情参考：https://cr.console.aliyun.com/cn-beijing/instances/mirrors
#### 1.2.2 用户公开镜像
* 详情参考：https://cr.console.aliyun.com/cn-beijing/instances/images
* All the images mentioned in this page are available in aliyun: [**Image List**](https://github.com/xiaotaw/Notes/blob/master/docker/images.md).

#### 1.2.3 将镜像上传至阿里云本地镜像仓库
* 登录[阿里云镜像服务管理控制台](https://cr.console.aliyun.com)，进入容器镜像服务
* 默认实例->访问凭证，设置固定密码
* 默认实例->命名空间，根据指引创建命名空间 xt-cuda
* 默认实例->镜像仓库，网页顶部左端，选择华北2(北京)，然后根据指引，在命名空间下，创建镜像仓库 cuda，不关联其他账号，使用本地镜像仓库，并设置为公开。
* 创建好的仓库->操作->管理中，参考操作指南，上传镜像。
* 登录Docker Registry: `docker login --username=xxxxxxxxxxx registry.cn-beijing.aliyuncs.com`，输入固定密码登录
* 将本地镜像标tag：`docker tag b400c26dd64a registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 上传至阿里云：`docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`




# 2 创建镜像
## 2.1 基于cuda:10.1-cudnn7-runtime-ubuntu16.04，创建pytorch1.1.0镜像
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

# 3 保存镜像
docker save -o xxx.tar IMAGE
