## 目录
* [GibsonEnv](#GibsonEnv)
* [参考资料](#参考资料)

## GibsonEnv
### GibsonEnv体验
```bash
# 官网介绍的docker方法
# docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset gibson
# 结合个人情况，稍作改版：文件映射一个大的范围，进入容器后，再用软连接的方式创建assert/dataset文件夹
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/xt/Documents/data/:/data/ registry.cn-beijing.aliyuncs.com/xt-cuda/gibson:0.3.1_tf_cpu_optimized
# 进入容器后
cd gibson/assets && rmdir dataset
ln -s /data/GibsonEnv/gibson/assets/dataset dataset
```

## 参考资料
1. 李宏毅深度强化学习(国语)课程(2018) https://b23.tv/av24724071/p1

