# 从官网获取指定镜像
* 浏览器登录[dockerhub](https://hub.docker.com/)，搜索CUDA，进入nvidia/CUDA
* 找到对应的docker pull command `docker pull nvidia/cuda`，直接使用将获取tag为latest的镜像
* 在overview页面，可以寻找最新镜像对应的tag，例如：10.1-cudnn7-runtime-ubuntu16.04
* 使用命令获取镜像：`docker pull nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 使用命令查看镜像：`docker images`    

REPOSITORY | TAG | IMAGE ID | CREATED | SIZE
-|-|-|-|-
nvidia/cuda | 10.1-cudnn7-runtime-ubuntu16.04 | b400c26dd64a | 7 days ago | 1.64GB


# 将镜像上传至阿里云本地镜像仓库
* 登录[阿里云镜像服务管理控制台](https://cr.console.aliyun.com)，进入容器镜像服务
* 默认实例->访问凭证，设置固定密码
* 默认实例->命名空间，根据指引创建命名空间 xt-cuda
* 默认实例->镜像仓库，网页顶部左端，选择华北2(北京)，然后根据指引，在命名空间下，创建镜像仓库 cuda
* 创建好的仓库->操作->管理中，参考操作指南，上传镜像。
* 登录Docker Registry: `docker login --username=xxxxxxxxxxx registry.cn-beijing.aliyuncs.com`，输入固定密码登录
* 将本地镜像标tag：`docker tag b400c26dd64a registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`
* 上传至阿里云：`docker push registry.cn-beijing.aliyuncs.com/xt-cuda/cuda:10.1-cudnn7-runtime-ubuntu16.04`
