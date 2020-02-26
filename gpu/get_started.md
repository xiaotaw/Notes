## 目录
* [简介](#写在前面)
* [NVIDIA显卡驱动安装](#NVIDIA显卡驱动安装)
  * [手动安装](#手动安装)
  * [自动安装](#自动安装)
* [CUDA安装](#CUDA安装)
  * [安装之前的考虑](#安装之前的考虑)
  * [下载并安装CUDA](#下载并安装CUDA)
  * [下载并安装cudnn](#下载并安装cudnn)
* [参考资料](#参考资料)



## 写在前面
1. 在深度学习和图像处理等计算密集型任务中，经常使用GPU等硬件进行加速运算。图形图像工作站中多使用A卡(AMD)，游戏和深度学习多实用N卡(NVIDIA)。
2. 以下内容基于：ubuntu操作系统，nvidia显卡

## NVIDIA显卡驱动安装
系统中一般自带一个基本的显卡驱动，但是功能非常少，发挥不出显卡的战斗力，需要替换成NVIDIA的驱动。安装方式可以选择手动，或者在ubuntu中可以使用命令行几行代码搞定。

### 手动安装

```bash
# 查看显卡信息，得到显卡型号为GeForce GTX 1080 Ti
lspci | grep -i vga
# 01:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1) 


# 查看操作系统，为linux 64位
uname -a
# Linux ubt 4.15.0-88-generic #88-Ubuntu SMP Tue Feb 11 20:11:34 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
```

1. 首先根据自身操作系统，和显卡型号，在nvidia官网下载驱动安装程序。https://www.nvidia.cn/Download/index.aspx?lang=cn。得到 NVIDIA-Linux-x86_64-440.59.run 安装包。

2. 安装驱动前，需关闭所有图形服务，并禁用已有的nouveau显卡驱动。
```bash
# 首先重启机器，进入命令行模式，这样会自动把所有图形应用和图形服务关闭。
sudo systemctl set-default multi-user.target 
sudo reboot

# 查看是否有系统中是否运行着自带的nouveau驱动，如有，按照以下步骤禁用。
lsmod | grep nouveau
# 禁用已有的驱动
sudo vi /etc/modprobe.d/blacklist-nouveau.conf
# 并在文件中添加如下内容
blacklist nouveau
options nouveau modeset=0
# 更新并重启系统
sudo update-initramfs -u
sudo reboot
# 确认nouveau已经被禁用
lsmod | grep nouveau
```
3. 安装NVIDIA显卡驱动，根据提示操作。
```bash
sudo bash NVIDIA-Linux-x86_64-440.59.run
```

4. 切回图形界面模式，重启系统(如不需要图形界面，可直接重启系统)
```bash
sudo systemctl set-default graphical.target
sudo reboot
```

5. 检测NVIDIA驱动是否安装好，使用nvidia-smi可获得显卡以及驱动版本信息。
```bash
nvidia-smi
```

### 自动安装
```bash
# 添加nvidia repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 查看适合本机的驱动版本
ubuntu-drivers devices
# driver   : nvidia-driver-390 - distro non-free
# driver   : nvidia-driver-430 - distro non-free
# driver   : nvidia-driver-435 - distro non-free recommended
# driver   : xserver-xorg-video-nouveau - distro free builtin

# 选择最新版安装
sudo apt install nvidia-driver-435

# 重启系统
sudo reboot

# 用nvidia-smi检查驱动安装情况
nvidia-smi
```

## CUDA安装

### 安装之前的考虑
1. 检查显卡是否支持CUDA。部分老的显卡可能不支持CUDA，可以在https://developer.nvidia.com/cuda-gpus查看。
2. CUDA版本的选择。基于CUDA开发的程序，以及依赖CUDA的第三方库如tensorflow或者pytorch，对CUDA版本都有一定的要求，需根据各方面的需求进行选择。
3. CUDA版本是否与驱动兼容。低版本的驱动，不支持高版本的CUDA，可在https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html查看

### 下载并安装CUDA
https://developer.nvidia.com/cuda-toolkit-archive，按照官网的安装指导，进行安装。


```bash
# 以下载runfile(local)为例，管理员权限运行安装程序，按照命令行的提示进行操作即可
sudo sh cuda_10.0.130_410.48_linux.run

# 最后检测是否安装成功，可看到cuda版本。
nvcc -V
```

### 下载并安装cudnn
官方指导 https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html， 下载前需注册，根据操作系统和CUDA版本，选择所需要的cudnn版本。下载后解压，将cudnn.h和cudnn.so复制到cuda相应的目录下即可。
```bash
# 假设下载的cudnn文件为 cudnn-10.2-linux-x64-v7.6.5.32.tgz
tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz

# 假设cuda安装路径为 /usr/local/cuda
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```




## 参考资料
[安装 nvidia 驱动，cuda, 以及 cudnn](https://www.jianshu.com/p/fc5edbd6f480)  
[Ubuntu 18.04 NVIDIA驱动安装总结](https://blog.csdn.net/tjuyanming/article/details/80862290)  
[CUDA官方安装指导](https://developer.nvidia.com/cuda-toolkit-archive)  
[cudnn官方安装指导](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)  


