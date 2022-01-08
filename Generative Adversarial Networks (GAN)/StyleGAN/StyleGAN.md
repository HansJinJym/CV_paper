# StyleGAN 相关技术梳理

## 图像生成
### StyleGAN
- 基于样式的生成网络，改进传统网络（ProGAN）的问题
- 创新点：W空间，特征解缠，AdaLIn仿射，常值输入，随机噪声
### StyleGAN2
- 改进水滴伪影问题，修改AdaLIN，引入调制解调器
### StyleGAN3
- 改进特征停滞问题
### SWAGAN
- 图像域的操作改为频率域，直接解决了高频问题
- 更少、更快的计算量
### MobileStyleGAN
- 利用小波、深度可分离卷积、蒸馏训练

## 动漫化
### CycleGAN
- 两个G两个D，解决风格迁移图像一一对应问题
### U-GAT-IT
- CycleGAN基础上，添加AdaLIN和attention

## 逆映射
### pSp
- 训练金字塔编码器，引入多种loss，将图像映射至潜码
### e4e
- 原理类似

## 属性编辑
### GANSpace
- 利用PCA（主成分分析）找到StyleGAN的潜码的可解释移动方向
### InterFaceGAN
- 通过线性变化潜码，学习一种解耦表征
### StyleSpace
- S空间解藕
### HiSD
- 层次性解藕，融合，达到编辑目的
### SeFa
- 对模型参数闭环求解
### StyleCLIP
- 语义控制编辑

## 图像动作驱动
### First-Order-Motion
- 通过关键点，计算变换图和掩盖图，送入解码器inpainting