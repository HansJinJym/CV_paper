# 3DDFA_v1
- Face Alignment in Full Pose Range: A 3D Total Solution
- 3D Dense Face Alignment (3DDFA)，对齐dense 3D Morphable Model (3DMM)

## Introduction
- 其他算法的问题
    - 只能针对小角度（<45度）的人脸对齐
- 问题原因分析
    - Modelling：丢失关键点信息和语义信息
    - Fitting：模型难以适应正脸到侧脸的各个角度
    - Training Data：现有数据集标注对不可见的关键点不友好
- 论文创新点
    - 采用拟合三维人脸来代替直接定位特征点，然后可以根据三维人脸进行特征点的标记
    - 设置新的级联卷积神经网络，并为该网络设置新的input_feature是PAF和PNCC的结合
    - 采用人脸轮廓合成方法处理300W数据集，从而得到300W-LP训练数据集，用该数据集训练模型，得到的模型结果比300W数据集的结果要好很多
    - 采用四维的四元数来代替欧拉角来表示旋转，从而避免了万向锁，同时将四元数除以根号f，将缩放量参数也合并到四元数中
    - 利用300W-LP数据集训练需要对比论文的模型，然后在同一数据集下进行测试该模型（AFLW与AFLW2000-3D与300W数据集），计算每个图片的NME，从而可以得到平均的NME，然后在大Table中进行比较各个论文的NME
- 主要解决问题
    - 为解决大姿态情况下人脸特征点丢失问题时，通过级联神经网络拟合三维人脸来代替直接标注特征点，之后再利用三维人脸信息可以再得到特征点的信息
    - 为解决如何拟合三维人脸模型的问题，设计了新的网络，并未该网络设置了新的input feature，同时设置了新的损失函数
    - 为解决训练数据集短缺的问题，提出了人脸轮廓合成方法，处理300w数据集得到300W-LP数据集

## 3DDFA
- Input Feature
    - 根据论文叙述，一般所有论文描述的特征图输入共有两种：第一种是直接将原图片作为特征图输入到神经网络中（imgae_view），第二种是按照模型的需要，将像素进行重排列后得到的特征图输入到神经网络中(model_view) 。本文对于特征图的设计采用了两种方法得到的两种特征图，然后将两种特征图进行结合。
    - PAF图的生成
    ![](https://img-blog.csdnimg.cn/995df4b9bdf04afc847308927ad5fc5d.png)
    - 大致步骤：首先初始化一个3DMM的参数P ，然后利用公式重建出三维人脸模型
        - 1：将该三维人脸模型进行上采样得到一个64x64的feature anchor （a）
        - 2：在将该三维人脸模型投影到二维平面图上，从而又得到一个64x64的feature anchor (b)，此时将两个feature anchor设置为一个可视化，一个不可视化
        - 3：分别剪切每个feature anchor大小为dxd，与原来的图片连接成扩展的二维图片（c）
        - 4：将该图片进行PAC卷积处理，得到64x64的PAF结果图(d) 
- NCC与PNCC
    - Projected Normalized Coordinate Code
    - 步骤
      - 1. 首先初始化一个3DMM的参数P ，然后利用公式重建出三维人脸模型维度是[3,53215]，第一行是代表该所有特征点的x轴坐标
      - 2. 将该三维人脸模型按照Eq.5进行归一化处理，得到NCC Fig.5a，此时用NCC作为该三维人脸模型的纹理颜色图
      - 3. 将三维人脸模型按照Eq.6投影到二维平面上，采用Z-BUFF算法渲染p2d，且NCC作为color_map，从而得到PNCC Fig.5b
- 网络结构
    - Fig.2
    - 网络输出的结果：在初始化参数$p$输入后，参数输出更新的数是$\Delta p$，此时参数的值是 $p+\Delta p$
- 数据集
    - 训练时主要采用人脸轮廓合成技术从300W数据集中合成300W-LP数据集。这个数据集的标签是每个图片真实的3DMM参数Param_GT，同时后面的实验结果证明，这个数据集对模型进行训练比300W数据集对模型进行训练的结果要优秀很多
- 损失函数及优化函数
    - 该论文主要使用vdc(顶点距离损失)，wpdc(权重参数距离损失)，vdc from wpdc（先用wpdc损失函数训练好模型后，在用wpdc训练好的模型参数初始化网络模型，此时继续用vdc损失函数微调神经网络模型），owpdc损失函数，主要使用了SGD优化函数 

# 3DDFA_v2
- Towards Fast, Accurate and Stable 3D Dense Face Alignment
- 解决3D密集人脸对齐中的速度、精度、稳定性问题
- CPU速度实时且准确率达到SOTA

## Introduction
- 大部分论文在进行人脸对齐重建的时候，大多数关注模型的精确性（模型的好坏，用NME来衡量），很大程度上忽略了模型的运行速度。所以该论文针对这一问题，在提高模型精确性的前提下，同时提高模型的运行速度。于此，在视频上进行三维人脸重建的领域越来越重要，而在视频上进行人脸对齐的稳定性问题仍然滞留，基于该问题提出了在视频上进行三维人脸重建提高稳定性的方法。
    - 1. 采用轻量级的网络模型回归出3DMM的参数，然后为该网络设置了meta-joint optimization优化策略，动态的组合wpdc和vdc损失函数，从而加速了拟合的速度，也使得拟合的效果更加精确
    - 2. 提出landmark-regression regularization（特征点回归正则化）来加速拟合的速度和精确度
    - 3. 为了解决在video上的三维人脸对齐任务（相邻帧之间的三维重建更加稳定，快速，连续性），在基于video数据上训练的模型，但video视频数据库缺乏时，提出了3D aided short-video-synthesis（三维辅助短视频合成技术），将一个静止的图片在平面内还有平面外旋转变成一个短视频

## Methodology
- 损失函数
    - vdc与wpdc
    - 设计fwpdc（快速权重参数距离损失），只加速fwpdc损失函数的收敛时间，但不改变收敛时的值。Algorithm.2
    - vanilla-joint optimization
    - meta-joint optimization
- 3D Aided Short-video-synthesis(三维辅助短视频合成技术)
    - 为了解决视频中三维人脸对齐的问题，基于视频数据训练的模型，由于视频数据非常稀缺，所以在此基础上提出了三维辅助短视频合成技术。该技术主要将一个静止的图片，利用平面内和平面外的旋转，从而将其转换为短视频
    - 经测试发现，该技术共有两个作用， 一是对模型的训练有促进作用（用NME来衡量模型是否优越，可用21，68，NK个三维特征点之间的距离损失来计算dist），二是促进了在视频上进行三维人脸重建时候的稳定性（用相邻帧之间的偏移差异来衡量视频中三维人脸重建的稳定性）

## Reference
[1] [Official repo v1](https://github.com/cleardusk/3DDFA)
[2] [Official repo v2](https://github.com/cleardusk/3DDFA_V2)
[1] [3DDFAv1](https://blog.csdn.net/aliuxuwhy/article/details/119394984)
[2] [3DDFAv2](https://blog.csdn.net/aliuxuwhy/article/details/120922716?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)