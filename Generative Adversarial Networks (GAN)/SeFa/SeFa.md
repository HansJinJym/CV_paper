# SeFa

- Closed-Form Factorization of Latent Semantics in GANs
- CVPR 2021 Oral
- 仅通过模型权重学习latent vector的偏移方向，研究如何通过无监督的方法操纵隐变量来改变生成图像

## Preliminaries
- G1映射：先将d维的latent vector，映射至m维的projected code y
- 传统的编辑方法都是直接对vector做操作，然后送入生成器G，得到新图像
- 传统有监督方法：
  - 1. 通过训练好的GAN合成大量的图像 
  - 2. 对合成图像进行手工标注（如人脸向左还是向右、汽车的大小）
  - 3. 通过标签训练一个线性分类器，调整隐变量，使其沿着垂直于分类边界的方向移动
  - 例如StyleGAN-Encoder，InterFaceGAN
- 有监督的方法需要清晰定义目标属性（比如发型就很难以二值来定义，也就无法训练线性分类器）并且需要手工标注，因此难以拓展。

## 无监督SeFa
- 对G1仿射模块操作。偏移本质：由Eq.3，模型权重中应当包含有偏移信息，因为偏移过程取决于参数A、偏移强度a、偏移方向n，偏移方向（图像修改的部分）全部包含在An中
- 因此可以通过分解权重A来得到latent directions
- n为方向向量，因此转置乘自己为1（相当于模长为1）。An为0时原图无变化，所以希望An越大，越大图像变化越明显。
- 将问题建模为Eq.4和Eq.5，其中4为寻找单一向量，5为寻找k组向量。Eq.4为寻找某一变换方向，Eq.5为寻找多组变换方向。
- 引入拉格朗日乘子后问题变为Eq.6，求导变为Eq.7，方程的解是寻找特征向量

## 融入至主流GAN网络
### PGGAN
- 由一组latent vector直接生成图像，因此SeFa直接学习latent code至feature map的转换
### StyleGAN
- 学习latent vector映射至style code
### BigGAN
- PGGAN和StyleGAN方式的结合