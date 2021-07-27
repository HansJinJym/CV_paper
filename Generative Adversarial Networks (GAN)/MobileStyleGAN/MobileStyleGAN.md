# MobileStyleGAN

### Info
- MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis
- 2021.04 ArXiv

### Intro
- 关注于StyleGAN2中计算复杂度最高的部分
- 原版StyleGAN2计算复杂，导致很难在移动端上使用
- 本文创新点：
  - 提出基于小波的CNN-GAN用于图像生成
  - 利用深度可分离卷积减小计算量
  - 引入一种改进的图像调制解调方式
  - 网络训练方式改为知识蒸馏法

### 预备知识(StyleGAN、模型加速、知识蒸馏、小波变换)
#### StyleGAN
- 逐级扩大分辨率与生成特征
- 输入改为常量，latent code并行控制特征
- 引入样式模块AdaIN，对仿射进行归一化操作
#### StyleGAN2
- 修正AdaIN，解决了水滴伪影问题
- 加入跳跃连接，解决跨级特征停滞问题[1]
- 平滑特征空间
#### StyleGAN2-ADA
- StyleGan2-ADA: Training Generative Adversarial Networks with Limited Data
- ArXiv 2020.06, NVLabs
- ADA: adaptive discriminator augmentation[2]
- 解决训练数据较少时，判别器过拟合的问题
#### 深度级可分离卷积
- Depthwise separable convolution
- Inception和MobileNet[3,4]
- 原版卷积核参数数量 3x3x3x4=108
- 逐通道卷积+逐点卷积 3x3x3+1x1x3x4=39
  ![](https://pic1.zhimg.com/80/v2-617b082492f5c1c31bde1c6e2d994bc0_1440w.jpg)
  ![](https://pic4.zhimg.com/80/v2-a20824492e3e8778a959ca3731dfeea3_1440w.jpg)
  ![](https://pic4.zhimg.com/80/v2-2cdae9b3ad2f1d07e2c738331dac6d8b_1440w.jpg)
#### 知识蒸馏
- ArXiv 2015.03，Distilling the Knowledge in a Neural Network
  - 知识蒸馏（Knowledge Distilling）是模型压缩的一种方法，是指利用已经训练的一个较复杂的Teacher模型，指导一个较轻量的Student模型训练，从而在减小模型大小和计算资源的同时，尽量保持原Teacher模型的准确率的方法
  - 模型压缩开山
- ArXiv 2019，Compressing GANs Using Knowledge Distillation
  - 最早的用蒸馏的办法解决GAN的压缩问题
- NIPS 2016，Packing convolutional neural networks in the frequency domain
  - 在频域中压缩神经网络
- AAAI 2020，Distilling portable Generative Adversarial Networks for Image Translation
  - 华为
  - 用蒸馏的办法解决GAN的压缩问题（Image Translation领域）
#### 小波变换
- ICLR 2019，Large Scale GAN Training for High Fidelity Natural Image Synthesis
  - DeepMind
  - BigGAN，4倍的参数量和8倍的batch size
  - 效果（IS、FID）比SOTA提升一倍
- ArXiv 2020.09，Not-So-Big-GAN: Generating High-fidelity Images on Small Compute with Wavelet-based Super-resolution
  - Not-So-Big-GAN
  - 作者证明基于小波的GAN效果会好于基于像素点的GAN
#### 图像频域
  ![](https://img-blog.csdnimg.cn/20190421235727836.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM4MDUyMzg0,size_16,color_FFFFFF,t_70)
  - 图像的频率是表征图像中灰度变化剧烈程度的指标，是灰度在平面空间上的梯度。
  - 对图像而言，图像的边缘部分是突变部分，变化较快，因此反应在频域上是高频分量；图像的噪声大部分情况下是高频部分；图像平缓变化部分则为低频分量。
  ![](https://img-blog.csdnimg.cn/20190810175954514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwODE1MjM3,size_16,color_FFFFFF,t_70)

### 网络结构
#### 图像频域优势
- 原始图像做Haar小波变换，Fig.2
- DWT比基于像素的原图包含更多结构信息，Fig.2
- DWT与IDWT可以去除乘法操作，全部变为加法，Eq.2（两个方向，每个方向一减一加，共四个）
- 作者认为，仅通过latent vector只能较好地生成低频特征（即低分辨率模块），不能对高分辨率模块进行较好的控制。转换至频域后，可以使原图中的高低频分量均得到同等控制。
#### 跨越连接修改
- StyleGAN2的prediction head是为了使更深层的网络更好训练
- 本文作者认为这个模块对生成结果没有什么影响，并且会增大计算量，因此修改为Fig.3
#### 卷积
- 全部改为深度级可分离卷积
- 原上采样+归一化操作改为：IDWT+1x1卷积用于训练 Fig.5, Fig.6
- 原版中间层RGB图像用于下一层的叠加，本文中间结果直接输出（3.2节，不是很懂）

### 训练细节
- 待完成




### 学习资料
- StyleGAN & StyleGAN2
  - [1. 知乎，网络详解](https://zhuanlan.zhihu.com/p/263554045)
  - [2. StyleGAN-ada](https://blog.csdn.net/KongCDY/article/details/117364818)
- 深度级可分离卷积
  - [3. 知乎，介绍](https://zhuanlan.zhihu.com/p/92134485)
  - [4. MobileNet](https://blog.csdn.net/c20081052/article/details/80703896)
- 小波变换与图像频域
  - [5. 图像频域介绍](https://blog.csdn.net/m0_38052384/article/details/89442510)