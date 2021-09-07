# CVPR2019 Pose Estimation

1. 3D Hand Shape and Pose from Images in the Wild 
   - [repo](https://github.com/boukhayma/3dhand)
   - python2.7 + pytorch0.3
    ![](https://github.com/boukhayma/3dhand/raw/master/pipeline.png)
   - 本文的亮点在于作者运用了一个reprojection，得到弱透视模型下的相机参数，使生成的MANO手掌模型可以投影至2D图像，并获得2D hand pose。如此便可以使用含有大量2D hand pose的数据集进行训练，以解决3D Hand-Object Pose数据集不足的问题。
    ![](https://img-blog.csdnimg.cn/20190727115809128.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NTY3ODUx,size_16,color_FFFFFF,t_70)
   - [参考](https://blog.csdn.net/qq_29567851/article/details/97499770)

2. CrossInfoNet: Multi-Task Information Sharing Based Hand Pose Estimation
   - [repo](https://github.com/dumyy/handpose)
   - python2.7
   - 从单张深度图估计手部姿态和关键点

3. 3D Hand Shape and Pose Estimation from a Single RGB Image
   - [repo1](https://github.com/3d-hand-shape/hand-graph-cnn/), [repo2](https://github.com/geliuhao/3DHandShapePosefromRGB)
    ![](https://github.com/3d-hand-shape/hand-graph-cnn/raw/master/teaser.png)
   - 环境
     - Ubuntu
     - python 3.7 + pytorch 0.4.1 + cpu
     - 服务器测试pytorch1.2.0，1.4.0，1.6.0，1.7.0均报错
     - 安装opendr时，pip安装报错，解决方案：
     ```shell
     sudo apt install libosmesa6-dev
     sudo apt-get install build-essential
     sudo apt-get install libgl1-mesa-dev
     sudo apt-get install libglu1-mesa-dev
     sudo apt-get install freeglut3-de
     ```
     - [参考](http://pythonheidong.com)
   - 论文
     - 输入：RGB图片，相机内参，手掌bbox
     - 输出：手掌Mesh + 3D Pose
     - 采用Graph CNN，得到3D hand mesh vertices
     - 现有的合成数据集大多只提供2D/3D手掌关节点的位置，没有包含3D Hand Shape的注释。论文作者合成STB数据集，不同手势、光、肤色、角度，共375k张RGB图像。
   - 效果
     - output文件夹
     - 2fps左右
     - [参考](https://blog.csdn.net/qq_29567851/article/details/97303494)

4. Self-Supervised 3D Hand Pose Estimation Through Training by Fitting
   - [repo](https://github.com/melonwan/sphereHand)
   - 基于深度图
   - 没开源

5. Pushing the Envelope for RGB-Based Dense 3D Hand Pose Estimation via Neural Rendering
   - 没开源

6. Monocular Total Capture:Posing Face, Body and Hands in the Wild
   - [repo](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture)
    ![](https://camo.githubusercontent.com/1b86af0a2937c53b6e284c6cd62ab46fd908a9dd7477daa788710630efe53b7e/68747470733a2f2f7869616e67646f6e676c61692e6769746875622e696f2f4d54435f7465617365722e6a7067)
