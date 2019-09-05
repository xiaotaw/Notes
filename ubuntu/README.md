## 目录
* [命令行/图形界面启动](#命令行/图形界面启动)
* [终端中文显示为问号](#终端中文显示为问号)
* [frp](#frp)
* [vim](#vim)
* [安装软件/程序/包](#安装软件/程序/包)
  * [ubuntu18.04安装libgdk2.0-dev报错](#ubuntu18.04安装libgdk2.0-dev报错)
  * [python安装opencv](#python安装opencv)

## 命令行/图形界面启动
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
## frp
* frp是一个内网穿透工具，需要一台有公网ip的服务器作为跳板。  
* 参考https://github.com/fatedier/frp

## vim
设置vim自动缩进，并且将tab替换为四个空格
```bash
# 打开vim配置文件，添加以下三行
$ vim ~/.vimrc
set ts=4
set expandtab
set autoindent
```

## 安装软件/程序/包
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

### python安装opencv
```bash 
conda install -c menpo opencv3
```

