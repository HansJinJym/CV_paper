# I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image
- ECCV 2020
- [Github](https://github.com/mks0601/I2L-MeshNet_RELEASE)
- image-to-lixel (line+pixel 线素) prediction network
-  以前大多数基于图像的3D人体姿态和Mesh估计的工作都是通过估计mesh模型的参数来实现，但是直接去回归参数是一种高度非线性的映射。
-  本文提出的 I2L-MeshNet （实现从图像到 lixel=线+像素的预测网络）则不是直接回归参数，而是来预测每个mesh顶点坐标在1D热图上的逐像素可能性

## Introduction
- 3D人体姿态和Mesh估计目的是在恢复3D人体关节点和Mesh顶点位置。由于复杂的人类关节和2D - 3D模糊性，这是一项非常具有挑战性的任务。SMPL 和 MANO 是应用最广泛的参数化人体和手部Mesh模型，分别可以代表各种人体姿态。最近3D人体姿态和Mesh估计的研究大部分基于模型来进行，从输入图像来估计SMPL/MANO的参数，另一部分则是基于无模型方法，直接估计Mesh顶点坐标，他们通过将Mesh模型中包含的联合回归矩阵乘以估计的Mesh来获得3D姿态。但是这些基于模型以及无模型的3D姿势和Mesh估计工作都破坏了输入图像中像素之间的空间关系，因为输出阶段的FC层。
- 为了不破坏空间关系，最新的3D姿态估计方法中，不是通过Mesh顶点来定位关节点坐标，而是利用热图来进行，**其中热图的每个值代表在输入图像的相应像素位置处的人体关节存在的可能性和深度值**。因此，它保留了输入图像中像素之间的空间关系并为预测不确定性来建模。
- 更准确地进行3D人体姿势和Mesh估计，本文将 I2L-MeshNet 设计为由 PoseNet 和 MeshNet 组成的级联网络。 其中 PoseNet 预测每个3D关节点坐标的基于lixel的1D热图；MeshNet 利用PoseNet的输出以及图像特征作为输入来预测3D Mesh顶点坐标的基于lixel的1D热图。由于人体关节的位置提供了有关人体网格顶点位置的粗略但重要的信息，因此将其用于3D的Mesh估计很自然，并且可以大大提高准确性。

## I2L-MeshNet
### Overall Pipeline
![](https://img-blog.csdnimg.cn/20201107104303566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNTIwNTk2,size_16,color_FFFFFF,t_70#pic_center)
- 由PoseNet和MeshNet极联组成

### PoseNet
-  PoseNet是来估计三个基于lixel的1D所有关节点的热图
![](https://img-blog.csdnimg.cn/20201108130042300.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNTIwNTk2,size_16,color_FFFFFF,t_70#pic_center)
- PoseNet用ResNet提取特征，然后上采样三次，特征图长宽均变为8倍，通道数由2048降至256
- z方向不是很理解

### MeshNet
- 结构与PoseNet基本相同

## Reference
[1] https://blog.csdn.net/qq_40520596/article/details/109531879
[2] https://blog.csdn.net/u011058765/article/details/116902572
