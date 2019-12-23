依赖tensorflow进行c/c++开发所需要的头文件和共享库文件，目前有：

tensorflow_c.so.r1.8.zip(下载链接见相关的README.md，在GLIBC2.12和GLIBCXX3.4.13环境下使用gcc-4.8编译，使用方法见test_c下的run.sh)

|-- r1.8
    |-- inc
        |-- tensorflow/
        ...
        |-- readme.txt
    |-- lib
        |-- linux64_cuda10.0_cudnn7.6_glibc2.12
            |-- libtensorflow_cc.so
            |-- libtensorflow_framework.so




tensorflow_c.so.r2.0.zip(下载链接见相关的README.md，在GLIBC2.12和GLIBCXX3.4.22环境下使用gcc-4.8编译，使用方法见test_c下的run.sh)
|-- r2.0
    |-- inc
        |-- external
        |-- tensorflow
        |-- third_party
    |-- lib
        |-- linux64_cuda10.0_cudnn7.6_glibc2.12_glibcxx3.4.22
            |-- libtensorflow_cc.so
