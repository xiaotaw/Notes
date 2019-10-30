## 目录
* [命令行or图形界面启动](#命令行or图形界面启动)
* [终端中文显示为问号](#终端中文显示为问号)
* [网络服务](#网络服务)
  * [frp](#frp)
  * [ss](#ss)
  * [vnc4server](#vnc4server)
  * [x11vnc](#x11vnc)
  * [端口映射](#端口映射)
* [本地服务管理](#服务管理)
  * [CodeServer](#CodeServer)
  * [本地服务管理命令](#本地服务管理命令)
* [vim](#vim)
* [vscode](#vscode)
* [查找动态库](#查找动态库)
* [安装软件-程序-包](#安装软件-程序-包)
  * [使用国内源](#使用国内源)
  * [pip使用清华源](#pip使用清华源)
  * [ubuntu18_04安装libgdk2.0-dev报错](#ubuntu18_04安装libgdk2.0-dev报错)
  * [ubuntu18_04安装VisualSFM](#ubuntu18_04安装VisualSFM)
  * [python安装opencv](#python安装opencv)
  * [安装nvidia显卡驱动](#安装nvidia显卡驱动)
  * [opencv和pcl](#opencv和pcl)

## 命令行or图形界面启动
* 有时候需要关闭图形界面（如：安装显卡驱动），可以通过设置命令行模式开机重启。
```bash
# 设置默认命令行模式启动(重启生效)
sudo systemctl set-default multi-user.target 

# 设置默认图形界面启动（重启生效）
sudo systemctl set-default graphical.target
```

## 终端中文显示为问号
```bash
# 进入容器，查看字符集
root@xxxxxxxxxxxx:/# locale
LANG=
LANGUAGE=
LC_CTYPE="POSIX"
LC_NUMERIC="POSIX"
LC_TIME="POSIX"
LC_COLLATE="POSIX"
LC_MONETARY="POSIX"
LC_MESSAGES="POSIX"
LC_PAPER="POSIX"
LC_NAME="POSIX"
LC_ADDRESS="POSIX"
LC_TELEPHONE="POSIX"
LC_MEASUREMENT="POSIX"
LC_IDENTIFICATION="POSIX"
LC_ALL=

# 查看容器支持的字符集
root@b18f56aa1e15:/# locale -a
C
C.UTF-8
POSIX

# POSIX不支持中文，更改为C.UTF-8即可（或者写入bashrc中）
export LANG=C.UTF-8

```
## 网络服务
### frp
* frp是一个内网穿透工具，需要一台有公网ip的服务器作为跳板。  
* 参考https://github.com/fatedier/frp

### ss
* 启动命令（老是忘记）：`sudo sslocal -c xxx.conf -d start`
* 参考http://tanqingbo.com/2017/07/19/Ubuntu%E4%BD%BF%E7%94%A8shadowsocks%E7%BF%BB%E5%A2%99/

### vnc4server
* 常用命令  
终止：`vnc4server -kill :2 `  
启动：`vnc4server -geometry 960x768 :2`  
* 参考网址  
https://linuxconfig.org/vnc-server-on-ubuntu-18-04-bionic-beaver-linux  
https://cloud.tencent.com/developer/article/1350304 

### x11vnc
```bash
# 手动启动，需要先设置passwd
x11vnc -auth guess -forever -loop -noxdamage -repeat -rfbauth PATH_TO_YOUR_HOME/.vnc/passwd -rfbport 5900 -shared -o x11vnc.log

# 如果不允许多个客户端连接，可将-shared改为-nevershared
x11vnc -auth guess -nevershared -forever -loop -noxdamage -repeat -rfbauth PATH_TO_YOUR_HOME/.vnc/passwd -rfbport 5900 -o x11vnc.log

# 增加日志文件
nohup x11vnc -auth guess -nevershared -forever -loop -noxdamage -repeat -rfbauth /home/xt/.vnc/passwd -rfbport 5900 -o `date '+%Y-%m-%d_%H-%M-%S'`.log  2>&1  > run.log &

```
* 客户端推荐：TightVNC，网址https://www.tightvnc.com/，比RealVNC占用带宽小，反应更快，显示更清晰。 如果是windows且没有安装软件的权限，推荐使用Java版本的Viewer

* 为什么使用ssh隧道：
1. VNC默认不加密，在客户端和服务端之间的数据，经过Internet网络传输，可能有被及截获的风险。
2. 使用ssh隧道，就是在Internet中，用ssh的方式，连接客户端和主机，由于ssh是加密的，原本明文传输的VNC数据，使用了SSH的外壳后，在Internet中传输是安全的。
3. 在外部看来，客户端和服务端之间只有ssh连接，ssh传输的内容是加密的，就算被截取，也无法破译，不知道传输的数据是什么内容。

* TightVNC Java Viewer使用ssh隧道的方法：
0. 前提：远程服务器已经开启了x11vnc服务，并监听5900端口
1. 打开TightVNC客户端，勾选 Use SSH tunneling
2. 在SSH Server，SSH Port，SSH User中填写SSH登录远程服务器所需要的信息。
2.1 在SSH Server中填写远程服务器（开启了x11vnc服务的主机）ip地址
2.2 在SSH Port中填写远程服务器的SSH端口（默认为22）
2.3 在SSH User中填写远程服务器的用户名（可以暂时不填）
3. 在Remote Host中填写localhost（127.0.0.1应该也一样，没试过），Port中填写5900（均为默认值）
4. 相当于先ssh的方式登录了远程服务器，然后再连接localhost的5900端口。



### 端口映射
（百度NAT）

## 服务管理

### CodeServer
```bash
docker run -it --rm --network="host" -v "${HOME}/.local/share/code-server:/home/coder/.local/share/code-server" -v "$PWD:/home/coder/project" codercom/code-server:v2 code-server --host 0.0.0.0  --port 5905
```

### 本地服务管理命令
```bash
# 编写脚本
sudo vi /lib/systemd/system/XXX.service

# 启动
sudo systemctl daemon-reload
sudo systemctl enable XXX.service
sudo systemctl start XXX.service
sudo reboot

# 取消
sudo systemctl disable XXX.service
sudo systemctl stop XXX.service

```

## vim
设置vim自动缩进，并且将tab替换为四个空格
```bash
# 打开vim配置文件，添加以下三行
$ vim ~/.vimrc
set ts=4
set expandtab
set autoindent
```
## vscode
vscode调试python脚本时报错raise RuntimeError('already started')

解决办法：
在.py文件头添加如下语句：
import multiprocessing
multiprocessing.set_start_method(‘spawn’,True)

参考资料：
https://blog.csdn.net/wangzi371312/article/details/92796320

## 查找动态库
```bash
# 查找是否安装了xxx
ldconfig -p | grep xxx
```


## 安装软件-程序-包
### 使用国内源
参考https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/更新后，可能部分软件的源需手动添加

1. nvidia-docker
参照https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
2. neo4j
```bash
wget -O - https://debian.neo4j.org/neotechnology.gpg.key |  apt-key add -
echo 'deb https://debian.neo4j.org/repo stable/' |  tee /etc/apt/sources.list.d/neo4j.list
```

### pip使用清华源
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

### ubuntu18_04安装libgdk2.0-dev报错
* 问题: 
```bash
# 查看ubuntu版本
$ cat /etc/issue
Ubuntu 18.04.3 LTS \n \l

# 安装
$ sudo apt-get install libgdk2.0-dev
(...)
The following packages have unmet dependencies:  
 libgtk2.0-dev : Depends: libpango1.0-dev (>= 1.20) but it is not going to be installed  
                 Depends: libcairo2-dev (>= 1.6.4-6.1) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.
```

* 解决方法
```bash
# 按照提示，手动递归安装需要的依赖，直至发现：
$ sudo apt-get install libfontconfig1-dev
The following packages have unmet dependencies:
 libfontconfig1-dev : Depends: libfontconfig1 (= 2.12.6-0ubuntu2) but 2.12.6-0ubuntu2.3 is to be installed
E: Unable to correct problems, you have held broken packages.

$ sudo apt-get install libfontconfig1
libfontconfig1 is already the newest version (2.12.6-0ubuntu2.3).

# 貌似是因为系统中已经安装了libfontconfig1较新的版本2.12.6-0ubuntu2.3，而libfontconfig1-dev需要的是老的版本2.12.6-0ubuntu2。于是指定版本进行安装解决。
$ sudo apt-get install libfontconfig1=2.12.6-0ubuntu2

# 解决冲突后，顺利安装
$ sudo apt-get install libfontconfig1-dev libxft-dev
$ sudo apt-get install libpango1.0-dev libcairo2-dev
$ sudo apt-get install libgtk2.0-dev
```
* 其他方案（未尝试）
libgtk2.0-dev依赖得部分库需要一个比较老的版本。参考https://blog.csdn.net/u014527548/article/details/80251046


### ubuntu18_04安装VisualSFM
* 参考文档
推荐：https://www.jianshu.com/p/cc0b548313e9     
[VisualSFM安装官方文档](http://ccwu.me/vsfm/install.html)   
[官方示例](http://www.10flow.com/2012/08/15/building-visualsfm-on-ubuntu-12-04-precise-pangolin-desktop-64-bit/)，ubuntu版本太老，很多坑，看看就好

* 参考文档中，提供了相关文件的百度网盘链接，文件提取： https://pan.baidu.com/s/1sGrw51m509PHguSEB4L5Ag 密码: yck6

* 环境：ubuntu18.04（非docker中），64位，已安装cuda9.0

* 编译时文件结构
```
|--vsfm
  |--VisualSFM_linux_64bit.zip
  |--vsfm
    |--bin
      |--VisualSFM （可执行文件，客户端）
      |--libsiftgpu.so （从SiftGPU编译得到）
      |--libpba.so （从pba编译得到）
      |--cmvs genOption pmvs2 （从CMVS-PMVS编译得到）
    |-- ...
  |--SiftGPU-V400.zip
  |--SiftGPU
    |-- ...
  |--pba_v1.0.5.zip
  |--pba
    |-- ...
  |--CMVS-PMVS.zip
  |--CMVS-PMVS
    |-- ...
```

* 编译过程
```bash
# 安装CUDA（由于已经安装，略过，具体参考官方文档https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/CUDA_C_Getting_Started_Linux.pdf）

# 安装依赖
$ sudo apt install make build-essential pkg-config liblapack-dev gfortran jhead imagemagick libc6-dev-i386 libgtk2.0-dev libdevil-dev libboost-all-dev libatlas-cpp-0.6-dev libatlas-base-dev libcminpack-dev libgfortran3 libmetis-edf-dev libparmetis-dev freeglut3-dev
如果遇到libgtk2.0-dev安装问题，参考[ubuntu18.04安装libgdk2.0-dev报错](#ubuntu18.04安装libgdk2.0-dev报错)

# 
$ cd && mkdir vsfm && cd vsfm

# 下载VisualSFM客户端，解压，编译
#如果提示“...cannot be used when making a PIE project, recompile with -fPIC”，打开makefile文件，在LIB_LIST += 后添加“-no-pie”，然后重新make。
$ wget http://ccwu.me/vsfm/download/VisualSFM_linux_64bit.zip
$ unzip VisualSFM_linux_64bit.zip
$ cd vsfm
$ make 
$ cd ..

# 下载SiftGPU，解压编译，
# 注：①官网链接下不了，自行找百度网盘链接下载SiftGPU-V400.zip；②github上的SiftGPU不一定匹配（未尝试）
# 编译期间出现两个问题，一是gcc/g++版本7.x过高，按照提示，软连接到gcc-5/g++-5；二是依赖于GLWE，sudo apt-get install libglew-dev安装解决。
$ unzip SiftGPU-V400.zip
$ cd SiftGPU 
$ make
$ cp bin/libsiftgpu.so ../vsfm/bin/
$ cd ..

# 下载pba，解压编译
# 注：① 修改src/pba/SparseBundleCU.h和pba.h，分别在开头加上 #include <stdlib.h>
#    ② make不能多线程，makefile中，driver依赖于pba
$ wget http://grail.cs.washington.edu/projects/mcba/pba_v1.0.5.zip
$ unzip pba_v1.0.5.zip
$ cd pba
$ make
$ cp bin/libpba.so ../vsfm/bin/
$ cd ..

# 编译生成pmvs2，cmvs，genOption
# 从百度网盘下载的压缩包
$ unzip CMVS-PMVS.zip && cd CMVS-PMVS/program
$ mkdir build && cd build
$ cmake ..
$ make
$ cp main/pmvs2 ../../../../vsfm/bin
$ cp main/cmvs ../../../../vsfm/bin
$ cp main/genOption ../../../../vsfm/bin
cd ../../../..
```
* 最后vsfm/bin/VisualSFM即为可执行文件，运行可得到客户端。


### python安装opencv
```bash 
conda install -c menpo opencv3
```

### 安装nvidia显卡驱动
```bash
# 参考https://www.jianshu.com/p/7373be733226

# 去除已有的显卡驱动，禁用默认的显卡驱动
# 进入multi-user模式，命令行模式，禁用图形界面服务


# 检测系统适合的显卡
ubuntu-drivers devices

# 选择系统推荐的驱动，进行安装
sudo apt-get install nvidia-driver-430
```

### opencv和pcl
```bash
# opencv库
sudo apt-get install libopencv-dev
# pcl库和pcl viewer
sudo apt-get install libpcl-dev pcl-tools
```
