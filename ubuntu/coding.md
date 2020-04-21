## 目录
* [环境配置](#环境配置)
  * [msys2](#msys2)
* [C/C++](#ccplusplus)
  * [json](#json)
  * [std::string.c_str](#std::string.c_str)
  * [Unix Domain Socket](Unix Domain Socket)

**说明：和编程相关内容慢慢转移至本文件下**

## 环境配置
### msys2
#### 简介
在linux环境下编译c/c++程序有gcc/g++，make，cmake等一系列工具，使用非常方便。msys2是为windows系统构造了一个类似linux shell的工具，使得在windows上也能享受这种便利，编译生成的可执行文件（exe，dll）能再windows上运行。

msys for 32 bit OS, msys2 for 64 bit OS.

#### 安装 
能够科学上网的话，可以访问[msys2官方网站](www.msys2.org)下载安装；不能科学上网，下载会很慢，建议使用镜像站。  
镜像站列表：[ustc](https://lug.ustc.edu.cn/wiki/mirrors/help/msys2)，等。  

使用ustc镜像站：请访问该镜像目录下的 distrib/ 目录（x86_64、i686），找到名为 msys2-<架构>-<日期>.exe 的文件（如 msys2-x86_64-20141113.exe），下载安装即可。

#### pacman配置
```vim
#编辑 /etc/pacman.d/mirrorlist.mingw32 ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/mingw/i686

#编辑 /etc/pacman.d/mirrorlist.mingw64 ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/mingw/x86_64

#编辑 /etc/pacman.d/mirrorlist.msys ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/msys/$arch

#然后执行 pacman -Sy 刷新软件包数据即可。
```

之后又检查了一下，用清华大学开源镜像站比较快。

#### 参考资料
1. [ustc MSYS2 镜像使用帮助](https://lug.ustc.edu.cn/wiki/mirrors/help/msys2)  
2. [msys 的安装和使用](https://blog.csdn.net/brooknew/article/details/86472420)  
3. [msys2的使用](https://blog.csdn.net/ldpxxx/article/details/87977089)  
4. [thu MSYS2 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/msys2/)


## ccplusplus
### json
传说中c++最好用的json库，[nlohmann/json](https://github.com/nlohmann/json)

#### 问题
1. 貌似和python的json不太一样  
2. 按照参考资料1中的说法，仅支持utf-8文件

#### 参考资料
1. csdn blog [C++】nlohmann json包读取json文件异常的错误](https://blog.csdn.net/kuyu05/article/details/88561319?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
2. csdn blog [C++使用nlohmann json](https://blog.csdn.net/wphkadn/article/details/97417700)
3. csdn blog [c++类对象获得nlohmann::json配置的方便用法](https://blog.csdn.net/hongmaodaxia/article/details/95731211?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
4. github [nlohmann/json](https://github.com/nlohmann/json)


### std::string.c_str
使用std::string.c_str函数时，返回的是指向std::string内容的const char*，正常情况下对该内容进行修改都是不被允许的。  
但在某些情况下（比如gcc 4.9.2，使用char*做类型强制转换后），可以修改std::string内部的值。  
样例见example下的test_c_str.cpp。  


### Unix Domain Socket
之前有用过一个python版的UDS，在Notes/others/unix_domain_socket下

#### c版本server和client
代码见https://github.com/xiaotaw/odas下demo/tools

#### 参考资料
1. cnblogs [Unix domain socket 简介](https://www.cnblogs.com/sparkdev/p/8359028.html)
2. Linux C编程一站式学习[UNIX Domain Socket IPC](http://docs.linuxtone.org/ebooks/C&CPP/c/ch37s04.html)

