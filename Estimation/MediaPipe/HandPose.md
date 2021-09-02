# Hand Pose Estimation

## OpenPose
- [My Github](https://github.com/HansJinJym/CV_paper/tree/master/Estimation/OpenPose)， [Official Github](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- 自下而上的检验，先检验图像中全部关键点，再进行关键点连接
- 返回2D关键点坐标
- [官方3D人体关键点重建](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/3d_reconstruction_module.md)，只有一个demo，作者也不准备继续更新
![](https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/.github/media/openpose3d.gif)
- 人体3D关键点，using OpenPose with 3D baseline [3D-pose-baseline](https://github.com/ArashHosseini/3d-pose-baseline)，ICCV2017
![](https://github.com/ArashHosseini/3d-pose-baseline/raw/master/imgs/viz_example.png?raw=1)
- openpose(?) [bilibili](https://www.bilibili.com/video/av247273192/)
- Hand Keypoint Detection in Single Images using Multiview Bootstrapping
  ![](https://img-blog.csdnimg.cn/20181105181701135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE2ODE5NTI=,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20181106103148951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE2ODE5NTI=,size_16,color_FFFFFF,t_70)
  - 论文需要详细看，确认代码openpose是否有3D手部关键点检测功能

## AlphaPose
- [My Github](https://github.com/HansJinJym/CV_paper/tree/master/Estimation/AlphaPose)，[Official Github](https://github.com/MVIG-SJTU/AlphaPose)
- 自上而下的检验，先目标检测，再识别关键点
- 0.4.0版本包含了手部关键点检测，2D

## DensePose
- 全身，待调研

## MediaPipe
- [手部3D关键点](https://google.github.io/mediapipe/solutions/hands.html)
- [Github](https://github.com/google/mediapipe)
![](https://github.com/google/mediapipe/raw/master/docs/images/mobile/hand_tracking_android_gpu_small.gif)
- 谷歌开源多媒体机器学习框架
- 跨平台，支持嵌入式，移动设备，核心框架由c++实现
- 跑了一个demo，速度很快，约45fps，准确率较高
- 支持3D手部关键点检测及追踪，伪3D，xy坐标为图像绝对坐标，z坐标是相对坐标

### MediaPipe Hands
- MediaPipe Hands: On-device Real-time Hand Tracking
- two-stage: palm detector + hand landmark model
- BlazePalm Detector
  - 检测器任务较难，因为手部形变程度大，且不像脸部有固定的关键特征点
  - 因此先训练一个手掌检测器，利用非极大抑制，减少大量锚框
  - 编解码器结构
  - focal loss
- Hand Landmark Model
  - 根据palm结果继续预测2.5D关键点
  - 三个输出：21个2.5D关键点，图像有手的概率，左手还是右手
- 手掌检测器不会每帧都调用，只有首帧和追踪不到时才会调用，节省大量运算时间。利用前一帧对手掌的定位来确定当前帧的位置，当前帧手掌检测的概率低于阈值时重新检测手掌。Fig.5是完整检测流程。
- 应用：AR，手势识别等