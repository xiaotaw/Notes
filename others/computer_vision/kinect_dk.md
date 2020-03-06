## 目录
* [简介](#简介)
* [安装sdk](#安装sdk)
* [上手使用](#上手使用)
* [样例](#样例)
* [问题](#问题)

### 简介
1. Azure kinect dk只支持企业用户购买，比较麻烦。
2. PC环境为ubuntu18.04，恰好官方支持。

### 安装sdk

```bash
# add microsoft's package repository, follow instruction: 
# https://docs.microsoft.com/en-us/windows-server/administration/linux-package-repository-for-microsoft-software
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update

# install sdk by comandline
# https://docs.microsoft.com/en-us/azure/Kinect-dk/sensor-sdk-download
sudo apt install k4a-tools

# for latest version, refer to usage doc at github:
# https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
sudo apt install libk4a1.3-dev
```

### 上手使用
1. 参考[官方参考文档：快速入门](https://docs.microsoft.com/zh-cn/azure/Kinect-dk/set-up-azure-kinect-dk)。  
2. kinect dk连接电源线(圆孔) 。 
3. kinect dk连接数据线(typeC)，另一端需USB3.0连接至电脑，usb2.0不认。  
4. 连接完毕后，kinect dk前置LED流指示灯会闪一下，慢慢熄灭；后置LED指示灯稳定为白色，若一直在闪白色，连得可能是usb2.0。  
5. 打开sdk自带的k4aviewer，即会弹出一个可视化界面，可以可看到红外，彩色，深度三个图(如果前面的相机保护膜没撕下，图像是很模糊的)。  
6. 摄像头前面的红外发射灯工作，可以看到一个小红点(约3毫米长的扁平方形？看不太清)。  
7. 参考文档后续，熟悉kinect dk。

### 样例
example目录下有简单的使用样例(没优化，画面有点延迟感)

样例来源于样例来源于https://github.com/forestsen/KinectAzureDKProgramming中的OpenCV_OneKinect

### 问题
1. 深度图数据不完整  
1.1 物体边缘缺数据，
1.2 遮挡缺数据，红外发射，红外接收相机，以及彩色相机位置不重合，导致部分彩色相机可见，但是红外相机中为阴影（红外发射出的光，并没有照到）
1.3 红外吸收（黑色头发基本吸收了，但有些衣服上的黑色不吸收红外）
1.4 镜面反射，近距离的镜面（比如屏幕）反射的红外过强，会有一团圆形黑影
1.5 其他反射（墙角）
1.6 不知是不是镜头质量的问题，偶尔有黑色圆圈飘过。

总之，深度图各种问题，比预料中的差不少。

2. 深度图与彩色图匹配  



3. IMU数据的使用



