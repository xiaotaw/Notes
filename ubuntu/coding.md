## 目录

- [目录](#目录)
- [环境配置](#环境配置)
  - [msys2](#msys2)
- [ccplusplus](#ccplusplus)
  - [json](#json)
  - [std::string.c_str](#stdstringc_str)
  - [Unix Domain Socket](#unix-domain-socket)
  - [TimerLog](#timerlog)
  - [CoreDumped如何debug](#coredumped如何debug)
  - [性能分析工具Pref](#性能分析工具pref)
  - [查找动态库](#查找动态库)
- [Python](#python)
  - [tf dataset from generator](#tf-dataset-from-generator)
  - [ConstrainedLinearRegression](#constrainedlinearregression)
  - [pytest](#pytest)
  - [pip使用清华源](#pip使用清华源)
- [perl](#perl)
  - [perl正则表达式中使用非英文字符](#perl正则表达式中使用非英文字符)
- [shell](#shell)
  - [打包源文件](#打包源文件)
  - [shell里面判断字符串是否为空](#shell里面判断字符串是否为空)
- [clang-format和doxygen](#clang-format和doxygen)
- [ProblemSet](#problemset)

**说明：和编程相关内容慢慢转移至本文件下**

## 环境配置

### msys2

- 简介  
在linux环境下编译c/c++程序有gcc/g++，make，cmake等一系列工具，使用非常方便。msys2是为windows系统构造了一个类似linux shell的工具，使得在windows上也能享受这种便利，编译生成的可执行文件（exe，dll）能再windows上运行。msys for 32 bit OS, msys2 for 64 bit OS.

- 安装  
能够科学上网的话，可以访问[msys2官方网站](www.msys2.org)下载安装；不能科学上网，下载会很慢，建议使用镜像站。镜像站列表：[ustc](https://lug.ustc.edu.cn/wiki/mirrors/help/msys2)等。之后又检查了一下，用清华大学开源镜像站比较快。  
使用ustc镜像站：请访问该镜像目录下的 distrib/ 目录（x86_64、i686），找到名为 msys2-<架构>-<日期>.exe 的文件（如 msys2-x86_64-20141113.exe），下载安装即可。

- pacman配置

```vim
#编辑 /etc/pacman.d/mirrorlist.mingw32 ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/mingw/i686

#编辑 /etc/pacman.d/mirrorlist.mingw64 ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/mingw/x86_64

#编辑 /etc/pacman.d/mirrorlist.msys ，在文件开头添加：
Server = http://mirrors.ustc.edu.cn/msys2/msys/$arch

#然后执行 pacman -Sy 刷新软件包数据即可。
```

- 参考资料
  - [ustc MSYS2 镜像使用帮助](https://lug.ustc.edu.cn/wiki/mirrors/help/msys2)  
  - [msys 的安装和使用](https://blog.csdn.net/brooknew/article/details/86472420)  
  - [msys2的使用](https://blog.csdn.net/ldpxxx/article/details/87977089)  
  - [thu MSYS2 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/msys2/)

## ccplusplus

### json

- 传说中c++最好用的json库，[nlohmann/json](https://github.com/nlohmann/json)
- 问题
  - 貌似和python的json不太一样  
  - 按照参考资料1中的说法，仅支持utf-8文件
- 参考资料
  - csdn blog [【C++】nlohmann json包读取json文件异常的错误](https://blog.csdn.net/kuyu05/article/details/88561319?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
  - csdn blog [C++使用nlohmann json](https://blog.csdn.net/wphkadn/article/details/97417700)
  - csdn blog [c++类对象获得nlohmann::json配置的方便用法](https://blog.csdn.net/hongmaodaxia/article/details/95731211?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
  - github [nlohmann/json](https://github.com/nlohmann/json)

### std::string.c_str

> 使用std::string.c_str函数时，返回的是指向std::string内容的const char*，正常情况下对该内容进行修改都是不被允许的。  
但在某些情况下（比如gcc 4.9.2，使用char*做类型强制转换后），可以修改std::string内部的值。  
样例见example下的test_c_str.cpp。  

### Unix Domain Socket

- python版：之前有用过一个python版的UDS，在example_py/unix_domain_socket下
- c版本：server和client，代码见[https://github.com/xiaotaw/odas]下demo/tools
- 参考资料
  - cnblogs [Unix domain socket 简介](https://www.cnblogs.com/sparkdev/p/8359028.html)
  - Linux C编程一站式学习[UNIX Domain Socket IPC](http://docs.linuxtone.org/ebooks/C&CPP/c/ch37s04.html)

### TimerLog

1. 使用TimerLog，可以方便地跟踪各个步骤的耗时。
2. 使用类的静态成员，作为全局变量的替代
3. 代码见example/test_TimeLog.cpp

### CoreDumped如何debug

- `gdb <executable> core`

```bash
# 让linux系统生成core文件
# 运行`ulimit -c`，如果显示为0，则表明限制生成的core文件大小为0，即不生成core文件
ulimit -c unlimited

# debug模式编译程序
gcc -g -O0 test.cc

# 运行程序，core dumped之后，会在当前路径下生成core文件，如
./a.out

# gdb debug
gdb a.out core

# 可用bt查看函数调用信息
```

- valgrind，运行较慢

```bash
# 安装valgrind
apt-get install valrgind

# 内存检查
valgrind --tool=memcheck <exectuable> 
```

### 性能分析工具Pref

[https://perf.wiki.kernel.org/index.php/Main_Page]

### 查找动态库

```bash
# 查找是否安装了xxx
ldconfig -p | grep xxx
```

## Python

### tf dataset from generator

自定义dataset时，使用generator作为数据来源，示例见example_py/dataset_from_generator.py

### ConstrainedLinearRegression

- 简介：对参数进行限制的线性回归，示例见example_py/ConstrainedLinearRegression.py
- 测试：在python=3.6.9测试通过

```bash
# 创建python=3.6.9的环境
conda create -n clr python=3.6.9

# 执行conda activate clr，或者source activate clr，激活环境
conda activate clr

# 使用conda安装依赖sklearn
conda install h5py
conda install sklearn

# 或者使用pip安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple h5py
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn

# 测试
python3 ConstrainedLinearRegression.py
```

- 参考：[在Python上对每个系数有特定约束的多重线性回归](#https://www.pythonheidong.com/blog/article/166247/3ee8b193fa41e202a3e1/)，有部分修改  

### pytest

如果使用conda环境，可使用`conda install pytest`安装；
遇到报错`pytest: error: unrecognized arguments: --cov-report=html`，可`conda install pytest-cov`解决。

### pip使用清华源

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

## perl

### perl正则表达式中使用非英文字符

```vim
# 在perl脚本中，字符集为utf8
# 使用标准输入输出，且均为utf8

# 需要匹配阿拉伯数字，以及中文句号，且在两端加上空格
use Encode;

my $sen = "1你好啊。2你好啊。";

my $char = "。";

$sen =~ s/([\p{IsN}$char])/ $1 /g;

```

## shell

### 打包源文件

```bash
# 文件数目不多，可使用xargs
find . -name "*.h" -o name "*.c" | xargs tar zcvf src.tar.gz
# 文件数目多，先将文件名存入文件中，然后再打包
find . -name "*.h" -o name "*.c" > filename.list
tar zcvf src.tar.gz --files-from filename.list
```

### shell里面判断字符串是否为空

```bash
if [ -z $my_string ]; then
    echo "my_string is empty"
else
    echo "my_string is not empty"
fi
```

## clang-format和doxygen

参考[doxygen生成pdf文档](#https://blog.csdn.net/hahahaqwe123/article/details/107875776)

## ProblemSet

简单的题目，用于面试编程题，参见同目录下ProblemSet.md
