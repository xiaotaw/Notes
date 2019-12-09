## 目录
* [Virtual Env Simulator](#Virutal Env Simulator)
  * [GibsonEnv](#GibsonEnv)
  * [Habitat](#Habitat)
  * [数据集](#数据集)
* [AlphaGo](#AlphaGo)
* [参考资料](#参考资料)

## Virutal Env Simulator
### GibsonEnv
#### GibsonEnv体验
```bash
# 官网介绍的docker方法
# docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset gibson
# 结合个人情况，稍作改版：文件映射一个大的范围，进入容器后，再用软连接的方式创建assert/dataset文件夹
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/xt/Documents/data/:/data/ registry.cn-beijing.aliyuncs.com/xt-cuda/gibson:0.3.1_tf_cpu_optimized
# 进入容器后
cd gibson/assets && rmdir dataset
ln -s /data/GibsonEnv/gibson/assets/dataset dataset
```

### Habitat
Habitat是由Facebook维护。相比于Gibson，代码更规范，也更难阅读。宣称的优势是渲染速度能达到上千fps，比GibsonEnv高出一个数量级。

### 数据集
#### Gibson
1. 数据集大，三维重建质量一般
2. 需邮件申请

#### Matterport3D
1. 数据集大，三维重建质量一般
2. 需签署学术使用协议，邮件申请

#### Replica
1. 数据集小，三维重建质量好
2. 直接下载即可

#### SUNCG
1. 已下架

## AlphaGo
**论文地址: https://www.nature.com/articles/nature16961**

1. Input of Neural Network: 19*19*48。19*19是围棋棋盘，48个planes的特征，特征都是one-hot类型。
2. 注：后续AlphaGo Zero仅仅使用棋面信息，黑、白、空，没有多余的人工特征。

| Feature | # of planes | Description | 注释 |
|-|-|-|-|
| Stone color | 3 | Player stone / opponent stone / empty | 围棋棋盘状态，黑、白、空三种 |
| Ones | 1 | A constant plane filled with 1 | 不知道有什么用 |
| Turns since | 8 | How many turns since a move was played | 参考Liberties |
| Liberties | 8 | Number of liberties (empty adjacent points) | 论文描述：stone chain的气的数目，1,2,...,>=8气，分别为一个plane |
| Capture size | 8 | How many opponent stones would be captured | 参考Liberties |
| Self-atari size | 8 | How many of own stones would be captured | 参考Liberties |
| Liberties after move | 8 | Number of liberties after this move is played | 参考Liberties |
| Ladder capture | 1 | Whether a move at this point is a successful ladder capture | 是否征子成功 |
| Ladder escape | 1 | Whether a move at this point is a successful ladder escape | 是否征子失败 |
| Sensibleness | 1 | Whether a move is legal and does not fill is own eyes | 落子是否有效 |
| Zeros | 1 | A constant plane filled with 0 | 不知道有什么用 |
| -- | -- | -- | -- |
| Player color | 1 | Whether current player is black | 这不算在48个planes之中 |

围棋术语对照：
stone 棋子
move 一手，一着棋
liberties 气
capture 提子
ladder 征子
eye 围棋棋眼，眼位




## 参考资料
1. 李宏毅深度强化学习(国语)课程(2018) https://b23.tv/av24724071/p1

