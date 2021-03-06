## 目录
* [简介](#简介)
* [数学基础](#数学基础)
  * [四元数和三维旋转](#四元数和三维旋转)
  * [双四元混合Dual-Quaternion Blending](#双四元混合)
  * [李群和李代数](#李群和李代数)
  * [图优化](#图优化)
    * [数值优化](#数值优化)
    * [参考资料](#参考资料)
* [三维重建](#三维重建)
  * [相机模型](#相机模型)
  * [单目](#单目)
  * [对极几何](#对极几何)
  * [双目系统和sfm](#双目系统和sfm)
  * [Active and Volumetric Stereo](#ActiveAndVolumetricStereo)
  * [VisualSFM + Meshlab](#VisualSFM试用)
* [疑问](#疑问)
  * [怎么理解单应矩阵](#怎么理解单应矩阵)
* [参考资料](#参考资料)
  * [CS231A: Computer Vision, From 3D Reconstruction to Recognition](#CS231A)
  * [视觉slam十四讲](#视觉slam十四讲)
  * [sfm介绍博客](#sfm介绍博客)
  * [《计算机视觉中的多视几何》](#《计算机视觉中的多视几何》)

## 简介
记录一些sfm和slam的入门资料，以及学习过程中遇到的一些问题。

## 数学基础
### 四元数和三维旋转
http://www.bilibili.com/video/av33385105  
http://www.bilibili.com/video/av35804287 
cnblogs [四元数插值与均值（姿态平滑）](https://www.cnblogs.com/21207-iHome/p/6952004.html) 


### 双四元混合
[清晰易懂的CSDN博客：双四元混合DQB](https://blog.csdn.net/iosmichael/article/details/101417198)
[Dual Quaternion Blending论文](https://www.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf)

### 李群和李代数



### 图优化
#### 数值优化
https://github.com/xiaotaw/Notes/blob/master/computer_vision/numerical_optimization.md

#### 参考资料
一个帮助理解图优化的简单例子:https://blog.csdn.net/weixin_43540678/article/details/83831548

## 三维重建
### 相机模型
https://blog.csdn.net/AIchipmunk/article/details/48132109
https://github.com/xiaotaw/Notes/blob/master/computer_vision/cs231a_course_notes/01-camera-models.pdf
### 单目
https://github.com/xiaotaw/Notes/blob/master/computer_vision/cs231a_course_notes/02-single-view-metrology.pdf
### 对极几何
https://github.com/xiaotaw/Notes/blob/master/computer_vision/cs231a_course_notes/03-epipolar-geometry.pdf
### 双目系统和sfm
https://github.com/xiaotaw/Notes/blob/master/computer_vision/cs231a_course_notes/04-stereo-systems.pdf
### ActiveAndVolumetricStereo
https://github.com/xiaotaw/Notes/blob/master/computer_vision/cs231a_course_notes/05-active-volumetric-stereo.pdf
### VisualSFM试用
--todo


## 疑问
### 怎么理解单应矩阵
没完全明白单应矩阵


## 参考资料
* 推荐资料靠前
* [The Future of Real-Time SLAM and Deep Learning vs SLAM](http://www.computervisionblog.com/2016/01/why-slam-matters-future-of-real-time.html)需要科学上网
### CS231A
* CS231A: Computer Vision, From 3D Reconstruction to Recognition  
* 点评：非常清晰易懂，推荐看course_notes。不过需事先看一点csdn的博客，了解一些基本名词，问题的定义。
* 网站：http://web.stanford.edu/class/cs231a/course_notes.html  
* 其他：cs231a的课堂笔记3的中文翻译，对极几何 https://blog.csdn.net/Ketal_N/article/details/83744626

### 视觉slam十四讲
* https://www.bilibili.com/video/av59593514/?p1

### sfm介绍博客
* 博主aipiano的opencv实现，从sfm的原理到实现的代码介绍很详细，分为4个部分 https://blog.csdn.net/AIchipmunk/article/details/48132109

### 《计算机视觉中的多视几何》
* 评价：太难懂，不推荐
* 获取：腾讯云社区的一篇[文章](https://cloud.tencent.com/developer/news/274792)中提到，书籍《计算机视觉中的多视图几何》的百度云分享：https://pan.baidu.com/s/1glF0QaySRXd1cTZVtv5Kyg 密码：3jug

### 卡尔曼滤波
1. csdn blog[详解卡尔曼滤波原理](https://blog.csdn.net/u010720661/article/details/63253509)
2. 知乎[如何通俗并尽可能详细地解释卡尔曼滤波？ - 云羽落的回答](https://www.zhihu.com/question/23971601/answer/839664224)

