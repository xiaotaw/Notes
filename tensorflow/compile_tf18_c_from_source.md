

## 简介



## 编译
* 在centos6中，使用bazel编译c版本的tensorflow

* 因为生产环境中，GLIBC版本较低（大约是contos6.5，GLIBC版本为2.12），因此需自主编译tensorflow的动态库。

### docker创建系统环境
* contos6.5是非长期维护版本，这里使用centos6代替，各软件的支持较多，比如可以直接从nvidia/cuda获取contos6的镜像，但是没有contos6.5的。
* github上很多关于cuda10.1不兼容的issues，用cuda10.0的版本更合适。
```bash
# centos6 + cuda10 + cudnn7
docker pull nvidia/cuda:10.0-cudnn7-devel-centos6 

# -v /etc/localtime:/etc/localtime:ro 是为了容器内和宿主机时间同步，可以省略
# -v /home/xt/Documents/data/:/data/ 是为了方便数据保存备份，有些软件下载较慢，可以备份至宿主机，今后方便重用。
nvidia-docker run -it -v /home/xt/Documents/data/:/data/ -v /etc/localtime:/etc/localtime:ro nvidia/cuda:10.0-cudnn7-devel-centos6

# 以下均在容器内运行
```

* 查找并定位libc和libstdc++库

```bash
# 不更新一下，还找不到libc.so
yum update

ldconfig -p | grep libc.so
# 得到libc库的位置：/lib64/libc.so.6

ldconfig -p | grep libstdc++.so
# 得到libc库的位置：/usr/lib64/libstdc++.so.6
```

* 检查GLIBC和GLIBCXX的版本
```bash
strings /lib64/libc.so.6 | grep GLIBC
# 得知GLIBC版本最高支持2.12，比ubuntu18.04低（GLIBC_2.27）

strings /usr/lib64/libstdc++.so.6 | grep GLIBC
# 得知GLIBC版本最高支持3.4.13，比ubuntu18.04低（GLIBCXX_3.4.25）
```

* 用这种低版本的GLIBC和GLIBCXX编译出来的tensorflow动态库，兼容性应该很好。

### 编译安装bazel
* 不论是python版还是c/c++版，bazel是编译tensorflow的必备工具。bazel不同版本之间，兼容性有点问题。最好根据tensorflow的版本，选择合适的bazel的版本。

* 这里准备编译tensorflow1.8，对应使用的bazel版本为0.10.0，gcc版本为4.8。（参考编译python版tensorflow的案例，https://tensorflow.google.cn/install/source#tested_build_configurations）

* 首先是gcc
```bash
# 查看centos6默认gcc版本为4.4.7
gcc --version

# 安装一个额外的gcc4.8
# 参考https://blog.csdn.net/weixin_34384681/article/details/91921751
# 下载安装很慢，得很长一段时间。
yum install wget
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O \
     /etc/yum.repos.d/devtools-2.repo
yum install devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-gfortran devtoolset-2-gcc-c++

# 切换至gcc4.8，并查看gcc版本，得知gcc版本为4.8.2
scl enable devtoolset-2 bash
gcc --version

# 退出，至gcc4.4.7的版本
exit
```

* 其次bazel依赖于java
```bash
# 下载jdk 8的linux版，[oracle官网](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)下载不方便，这里提供一个百度网盘链接。
# 链接: https://pan.baidu.com/s/1NI7k_QYCXa8ZN9oQv7PA2w 提取码: 6zg9 复制这段内容后打开百度网盘手机App，操作更方便哦
# 从百度网盘中下载 jdk-8u172-linux-x64.tar.gz（手机下载速度较快）

tar jdk-8u172-linux-x64.tar.gz

# 简单配置java环境变量
export JAVA_HOME=`pwd`/jdk1.8.0_172
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# 也可以讲jdk放到/usr/local下,
mv jdk1.8.0_172 /usr/local
export JAVA_HOME=/usr/local/jdk1.8.0_172
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# 检查java
java -version
javac -version

# export只是临时添加环境变量；
# 更好的方式是，打开文件~/.bashrc，在末尾写入export语句，关闭保存，今后每次开启终端，都有java环境变量。
# source ~/.bashrc，使改动立即生效。
```


* centos6使用`yum install bazel`6默认安装的bazel版本可能过高，并且依赖高版本的GLIBC，不可行。
* 下载bazel的0.10.0版本的linux安装包，安装也失败。
* 最后采用源码编译bazel的方式。参考https://docs.bazel.build/versions/master/install-compile-source.html#bootstrap-bazel。
* 这里需注意的是，不能从github上直接下载源码，得从release中选择[0.10.0版本](https://github.com/bazelbuild/bazel/releases/tag/0.10.0)，选择下载[bazel-0.10.0-dist.zip](https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip)文件，解压后编译。

```bash
wget https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip
unzip bazel-0.10.0-dist.zip

cd bazel-0.10.0-dist

# 注意此时gcc需切换到4.8。
scl enable devtoolset-2 bash
bash compile.sh

# 编译得到output/bazel，bazel的路径添加至环境变量中，
export PATH=$PATH:`pwd`/output
which bazel

# 检查bazel版本为0.10.0
bazel version
```

### 准备tensorflow源码

* 准备1.8版本的tensorflow的源码
```bash
yum install git
# 从下载tensorflow源码
git clone https://github.com/tensorflow/tensorflow.git
# 切换到1.8版本
git checkout r1.8

# 在宿主机中已经提前下载好，通过docker的文件映射，放在容器/data路径下，可以省不少事。
```

### 安装tensorflow的依赖protobuf和eigen
* 安装依赖protobuf
* 参考文章：https://www.jianshu.com/p/d46596558640
```bash
# 安装automake和cmake
yum install autoconf automake libtool cmake

./tensorflow/contrib/makefile/download_dependencies.sh
# 下载不全没关系，protobuf和eigen下载了就行。

# protobuf
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure --prefix=/tmp/proto/
make -j8 && make install 


# eigen
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install
cd ../../../../../..

```

### 安装python等

```bash
# 
yum install centos-release-scl

yum install python27 python27-numpy python27-python-devel python27-python-wheel

### 编译
```

```
yum install git

yum install patch

```
# 报错: undefined reference to 'clock_gettime'
# 直接在bazel 命令中添加 --linkopt=-lrt 无效
# 参考：https://github.com/tensorflow/tensorflow/issues/15129
# 修改tensorflow/tensorflow.bzl，
def tf_cc_shared_object(
    name,
    srcs=[],
    deps=[],
    linkopts=[''],
    framework_so=tf_binary_additional_srcs(),
    **kwargs):
  native.cc_binary(
      name=name,
      srcs=srcs + framework_so,
      deps=deps,
      linkshared = 1,
      linkopts=linkopts + _rpath_linkopts(name) + select({
          clean_dep("//tensorflow:darwin"): [
              "-Wl,-install_name,@rpath/" + name.split("/")[-1],
          ],
          clean_dep("//tensorflow:windows"): [],
          "//conditions:default": [
              "-Wl,-soname," + name.split("/")[-1],
          ],
      }),
      **kwargs)
中的linkopts中添加'-lrt'，即：

def tf_cc_shared_object(
    name,
    srcs=[],
    deps=[],
    linkopts=['-lrt'],
    framework_so=tf_binary_additional_srcs(),
    **kwargs):
  native.cc_binary(
      name=name,
      srcs=srcs + framework_so,
      deps=deps,
      linkshared = 1,
      linkopts=linkopts + _rpath_linkopts(name) + select({
          clean_dep("//tensorflow:darwin"): [
              "-Wl,-install_name,@rpath/" + name.split("/")[-1],
          ],
          clean_dep("//tensorflow:windows"): [],
          "//conditions:default": [
              "-Wl,-soname," + name.split("/")[-1],
          ],
      }),
      **kwargs)

```

https://github.com/tensorflow/tensorflow/issues/15129
```














