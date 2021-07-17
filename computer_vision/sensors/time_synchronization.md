# Time Synchronization

## 目录
* [测试GNSS设备](#测试GNSS设备)
* [测试lidar设备](#测试lidar设备)
* [gnss给lidar授时](#gnss给lidar授时)
* [phase lock for velodyne VLP16](#phase lock for velodyne VLP16)



## 测试GNSS设备

### 准备工作
* GNSS接收机：华远星通HY-269（内置novatel 718d板卡，附带pwr线缆x1、IO线缆x1、com线缆x2、USB线缆x1、eth线缆x1，另需准备220V交流转9-30V直流适配器）  
* GPS天线：GPS500 三系统七频外置测量天线（附带天线线缆x2）  
* windows笔记本
* 示波器

### GNSS连接测试-USB
* 连接：GNSS接收机COM3（出厂COM3已被配置成USB连接） --- lemo转USB线缆 --- 笔记本
* 笔记本上安装NovAtel USBDriver、NovAtelConnect，软件和驱动可从[novatel](https://novatel.com/support/support-materials/software-downloads)官网下载
* 打开 NovatelConnect，可以看到几个COM口，连接任意一个（如COM6，注此处COM6为笔记本上的，与GNSS设备上的COM序号作区分），将GPS天线放在窗外（或室外），等待约40s（冷启动），即可看到收星、经纬度时间等信息。

![NovAtelConnect](https://github.com/xiaotaw/Notes/blob/master/computer_vision/pic/NovAtelConnect.jpg)

### GNSS连接测试-COM2 使用串口调试助手配置串口COM2，COM2连接测试
* USB连接成功后，可以配置其他串口
* 笔记本上打开串口调试软件（链接：[https://pan.baidu.com/s/1qgqDnSreAwtVSdEz_9jWdQ] 提取码：ocbf）
* 连接一个COM（如COM6），波特率9600
* 取消接收区的16进制的勾选，取消发送区16进制的勾选。
* 在发送区发送命令，`log com2 gprmc ontime 1`, `SERIAL CONFIG com2 9600`, `saveconfig`，注意，每个命令后都需带回车。其中为了给雷达授时，gprmc信息频率设置为1Hz，与pps信号周期相同。
* 连接：GNSS接收机COM2 --- lemo转db9线缆 --- 笔记本
* 笔记本上打开NovAtelConnect，可看到另外几个COM口，连接任意一个即可，同样可以看到收星、经纬度信息。

![SerialConnect](https://github.com/xiaotaw/Notes/blob/master/computer_vision/pic/SerialConnect.jpg)

### 示波器检测PPS和GPRMC
* **从IO口引出PPS**，接收机上IO口为lemo，对应的线缆为io线缆，一端为7芯的lemo连接器，一端为散线。根据该型号的GNSS接收机的使用手册，IO口中7号针脚为信号地线、2号针脚为pps，对应线缆分别为棕色和绿色，可以用万用表测连通来检验。
* **从COM2引出GPRMC**，接收机上COM2通信线缆，一端为5芯lemo连接器，一端为DB9F母头（减掉）。根据手册，COM2口中，2号针脚即可引出GPRMC，5号针脚为信号地。
* **示波器检测信号**，示波器表笔的笔尖接信号，夹子接信号地；调节示波器，可看到周期为1s的pps和gprmc信号；捕获1s，放大后可看到脉冲信号（黄色，波谷状）后，紧跟gprmc信号（绿色）。
* **设置PPS为正脉冲**，在串口调试工具中，`PPSCONTROL enable positive 1 2000`，`saveconfig`。`negative`为负脉冲，`positive`为正脉冲。

![oscillograph-negative](https://github.com/xiaotaw/Notes/blob/master/computer_vision/pic/oscillograph-negative.jpg)
![oscillograph-positive](https://github.com/xiaotaw/Notes/blob/master/computer_vision/pic/oscillograph-positive.jpg)

## 测试lidar设备
（todo：VLP16 lidar基本使用）

## gnss给lidar授时
参考资料：[https://www.linkedin.com/pulse/%E5%85%B3%E4%BA%8Evelodyne-lidar%E7%9A%84%E6%97%B6%E9%97%B4%E5%90%8C%E6%AD%A5-wei-weng]

GNSS设备引出的pps接VLP16的GPS pulse，GPRMC接GPS recieve。
笔记本打开雷达ip地址（默认192.168.1.201），可以看到GPS一栏中有信息，PPS一栏中为Locked。


## phase lock for velodyne VLP16
phase lock防双雷达干扰。

* 连接lidar 1：将lidar 1连接至电脑；修改网络连接，本机ip为192.168.1.13，子网掩码255.255.255.0，网关192.168.1.1；浏览器打开lidar默认ip地址192.168.1.201，检查gps和pps；设置默认ip为192.168.2.202，saveconfig；重启lidar，修改网络连接，本机ip为192.168.2.13，子网掩码不变，网关192.168.2.1；浏览器打开192.168.2.202，确认无误；修改data port为2369，修改Telemetry Port为8309。
* 连接lidar 2：将lidar 2连接至电脑；修改网络连接，本机ip为192.168.1.13，子网掩码255.255.255.0，网关192.168.1.1；浏览器打开lidar默认ip地址192.168.1.201，检查gps和pps。
* 启动ros驱动，`roslaunch VLP_points_2.launch`。注：文件[VLP16_points_2.launch](https://github.com/xiaotaw/Notes/blob/master/computer_vision/VLP16_points_2.launch)中的部分参数与前面设置的一致，双雷达launch文件参考[https://github.com/ros-driver/velodyne/issues/108]。


