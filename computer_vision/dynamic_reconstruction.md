## 目录 




### 符号约定
**粗体** 表示向量或矩阵，非粗体一般表示标量，比如三维点坐标为![](https://latex.codecogs.com/gif.latex?\mathbf{P})，相机内参矩阵![](https://latex.codecogs.com/gif.latex?\mathbf{K})。  
![](https://latex.codecogs.com/gif.latex?\textbf{P}=[P_x,P_y,P_z]^T)

** *手写体* ** 表示集合，比如RGB图像![](https://latex.codecogs.com/gif.latex?\mathcal{I}(u,v)) ，深度图![](https://latex.codecogs.com/gif.latex?\mathcal{D}(u,v)) 。

**哥特体** 表示李代数，例如表示三维空间旋转和平移的特殊欧式群![](https://latex.codecogs.com/gif.latex?\mathbf{SE}(3))，其对应的李代数为![](https://latex.codecogs.com/gif.latex?\mathfrak{se}(3))  

**下标** 表示索引或者维度，比如深度图的第u列第v行的元素![](https://latex.codecogs.com/gif.latex?\mathcal{D}_{u,v})，点![](https://latex.codecogs.com/gif.latex?\mathbf{P})在x, y, z轴的坐标值分别为![](https://latex.codecogs.com/gif.latex?P_x,P_y,P_z)

**注：** 在不引起歧义的情况下，三维点本身P和它的坐标![](https://latex.codecogs.com/gif.latex?\mathbf{P})通用，即![](https://latex.codecogs.com/gif.latex?P_x=\mathbf{P}_x)

### 问题提出
固定的RGBD相机，对一个动态场景以30ms/帧的频率连续拍摄，每一帧是分辨率为640x480的RGB彩色图
![](https://latex.codecogs.com/gif.latex?\mathcal{I}\in\mathbb{R}^{3\times640\times480})和深度图![](https://latex.codecogs.com/gif.latex?\mathcal{D}\in\mathbb{R}^{640\times480})。


相机内参矩阵![](https://latex.codecogs.com/gif.latex?\textbf{K}\in\mathbb{R}^{3\times3}) 已知，对于深度图中的每一个像素d=![](https://latex.codecogs.com/gif.latex?\mathcal{D}_{u,v})，根据投影关系，可计算出在三维空间中对应点的坐标**P** 。

![](https://latex.codecogs.com/gif.latex?P_z\begin{bmatrix}u\\\\v\\\\1\end{bmatrix}=\textbf{KP}=\begin{bmatrix}f_x&0&c_x\\\\0&f_y&c_y\\\\0&0&1\end{bmatrix}\space\begin{bmatrix}P_x\\\\P_y\\\\P_z\end{bmatrix})

注：![](https://latex.codecogs.com/gif.latex?\mathbf{P}_z=d=\mathcal{D}_{u,v})

将![](https://latex.codecogs.com/gif.latex?\mathcal{D})中每一个像素都计算相应的空间点，其颜色由![](https://latex.codecogs.com/gif.latex?[r,g,b]^T=\mathcal{I}_{u,v}) 给出，得到三维空间中的稠密点云，是当前场景在相机角度下的可视表面。

![点云示意图]()

**目标是：通过计算连续两帧之间物体的运动和表面的形变，将不同的物体/具有不同材质特性的物体分开。**


### 相关工作

#### 迭代最近邻(ICP)求解刚体平移和旋转

刚体P的平移和旋转变换可用矩阵来表示，![](https://latex.codecogs.com/gif.latex?\mathbf{T}\in\mathbf{SE}(3))，即特殊欧式群中的每一个元素，均可表示一个变换。

![](https://latex.codecogs.com/gif.latex?\mathbf{P'}=\mathbf{TP}=\begin{bmatrix}\mathbf{R}&\mathbf{t}\end{bmatrix}\mathbf{P})   
其中![](https://latex.codecogs.com/gif.latex?\mathbf{R}\in\mathbb{R}^{3\times3})表示旋转，![](https://latex.codecogs.com/gif.latex?\mathbf{t}\in\mathbb{R}^3)表示平移。（其中省略了齐次坐标转换）

矩阵直观，方便运算，但是不够紧凑。用![](https://latex.codecogs.com/gif.latex?\mathbf{SE}(3))对应的李代数![](https://latex.codecogs.com/gif.latex?\mathfrak{se(3)}) 表示。

两者转换关系为:  
![](https://latex.codecogs.com/gif.latex?\mathbf{SE}(3)=\mathbf{Exp}(\mathfrak{se}(3)))  
![](https://latex.codecogs.com/gif.latex?\mathfrak{se}(3)=\mathbf{Log}(\mathbf{SE}(3))) 

对于场景中刚体，假设其在相邻两帧之间的运动为![](https://latex.codecogs.com/gif.latex?\xi\in\mathfrak{se(3)})。

对于刚体上的每一点P，均有:  
![](https://latex.codecogs.com/gif.latex?\mathbf{P}_{t+1}=\mathbf{T}\mathbf{P}_t=\mathbf{Exp}(\xi)\mathbf{P}_t) 

通过高斯牛顿法，最小化“点-点”距离可求出  
![](https://latex.codecogs.com/gif.latex?\xi^*=\mathop{\arg\min}_{\xi}(\sum_P(\mathbf{P}_{t+1}-\mathbf{Exp}(\xi)\mathbf{P}_t)^2)) 

实际上我们并不知道两帧之间点P的精确对应关系，因此需要对上述公式做修改，用“点-面”距离代替“点-点距离”。  
![](https://latex.codecogs.com/gif.latex?\xi^*=\mathop{\arg\min}_{\xi}(\sum_P(\mathbf{P'}_{t+1}-\mathbf{Exp}(\xi)\mathbf{P}_t)\cdot\mathbf{n}_{t}))   

其中![](https://latex.codecogs.com/gif.latex?P'_t+1)为靠近![](https://latex.codecogs.com/gif.latex?P_t)的一个点，![](https://latex.codecogs.com/gif.latex?n_t)为![](https://latex.codecogs.com/gif.latex?P_t)处的法向量。



由于相邻两帧之间刚体位移较小，可以认为：同一个像素坐标(u,v)在相邻两帧![](https://latex.codecogs.com/gif.latex?\mathcal{D}_{t+1})和![](https://latex.codecogs.com/gif.latex?\mathcal{D}_t)对应的三维点![](https://latex.codecogs.com/gif.latex?P'_t+1)和![](https://latex.codecogs.com/gif.latex?P_t)足够靠近。


#### 动态场景

已有的解决方案1：假设![](https://latex.codecogs.com/gif.latex?\xi)是定义在时间和空间[x,y,z,t]上的函数（运动场？）。给定空间范围，假设为相机面前3m * 3m * 3m的立方体，将空间切割成3mm * 3mm * 3mm的小立方体，一共有1e9个小立方体，每个小立方体认为是刚体。同时为了降低运算量，不会计算每一个小立方体的![](https://latex.codecogs.com/gif.latex?\xi)，而是通过稀疏地采样得到一组小立方体（例如5000个）作为基础节点，其他的小立方体通过插值得到。同时对基础节点加一层约束，使得同一基础节点周围的点的形变尽量接近。（论文dynamic fusion）

#### 其他问题

1. 遮挡问题：可以先不考虑，假设我们关注的重建对象，在每一帧中都可见，且不被其他物体遮挡，简化问题，之后再拓展；当然物体的背面是看不见的，被自身遮挡了。    
2. 图像识别：图像识别可以识别出“人”“杯子”等类别。输入一张彩色图片，输出为一个或多个矩形框（左上、右下坐标表示）以及框内物体的类别。    


### 设想（还没整清楚）
假设三维空间中存在若干物体，我们用表面S来描述它们：（为什么不用小立方体的形式，小立方体数目太多了，还必须实现限定空间范围）

在动态的场景中，S随时间会变化，不同物体表面的变化规律应当不一样。现在不知道有统一的形式（参数化表示？），来描述这个表面，通过连续两帧之间的约束，求解表面的变化。

（相机每次拍摄只能得到物体表面的一部分，不是全部。）

最后，通过对这些变化进行分类/聚类，可以得到物体的材质。这个可以作为图像识别的一个补充，比如视频中到底是真人，还是蜡像人。

### 曲面参数化教程
写的非常好，http://sites.fas.harvard.edu/~cs277/handouts/param.pdf
