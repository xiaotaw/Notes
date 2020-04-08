## 目录

### user namespace



#### reference
1. [Docker容器中用户权限管理](https://www.cnblogs.com/zhouzhifei/p/11557118.html)    
2. [docker的user namespace功能](https://blog.51cto.com/yangzhiming/2384688)    
3. [ssh 直接登录docker容器](http://www.fecmall.com/topic/592)

### 备忘
docker run --runtime=nvidia -itd -v /data/DATASETS/ASR/:/data/DATASETS/ASR/ --name w2l wav2letter/wav2letter:cuda-latest

docker run --runtime=nvidia -itd -p 2222:22 -v /data/DATASETS/ASR/:/data/DATASETS/ASR/ --name w2lssh wav2letter/wav2letter:cuda-ssh-20200407 /usr/sbin/sshd -D
