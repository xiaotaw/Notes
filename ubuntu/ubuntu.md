## 目录
* [命令行/图形界面启动](命令行/图形界面启动)
* [一些BUG](一些BUG)

## 命令行/图形界面启动
* 有时候需要关闭图形界面（如：安装显卡驱动），可以通过设置命令行模式开机重启。
```bash
# 设置默认命令行模式启动(重启生效)
sudo systemctl set-default multi-user.target 

# 设置默认图形界面启动（重启生效）
sudo systemctl set-default graphical.target
```

## 一些BUG
### ubuntu18.04安装libgdk2.0-dev报错
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

