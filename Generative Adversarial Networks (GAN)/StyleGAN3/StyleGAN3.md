# Alias-Free Generative Adversarial Networks

- NVIDIA
- NeurIPS2021
- [介绍](https://www.thepaper.cn/newsDetail_forward_14881868)
- [ArXiv2106](https://arxiv.org/pdf/2106.12423.pdf)
- Abstract
  - 层次性生成图像的像素坐标会出现“粘滞”现象
  - 作者认为原因在于careless signal processing
  - 修改后的网络与StyleGAN2的FID相近，但是解决了上述问题
  - StyleGAN3未来可以用于视频和动画的生成
![](https://imagepphcloud.thepaper.cn/pph/image/158/320/891.gif)
![](https://imagepphcloud.thepaper.cn/pph/image/158/320/905.gif)
![](https://imagepphcloud.thepaper.cn/pph/image/158/320/925.gif)
![](https://imagepphcloud.thepaper.cn/pph/image/158/320/933.gif)

## Introduction
- "texture Sticking"问题，图一
- 图一左，皮毛清晰代表粘滞越强
- 图一右，interpolation形成光带代表粘滞性强
- Aliasing 混叠
  - 奈奎斯特准则
  ![](https://bkimg.cdn.bcebos.com/pic/4bed2e738bd4b31cc1c0638a8dd6277f9e2ff807?x-bce-process=image/resize,m_lfit,w_896,limit_1/format,f_auto)
  ![](https://bkimg.cdn.bcebos.com/pic/d62a6059252dd42aaebdf69a093b5bb5c9eab84f?x-bce-process=image/resize,m_lfit,w_876,limit_1/format,f_auto)
- contributions
  - Current upsampling filters are simply not aggressive enough in suppressing aliasing, and that extremely high-quality filters with over 100dB attenuation are required.
  - We present a principled solution to aliasing caused by pointwise nonlinearities by considering their effect in the continuous domain and appropriately low-pass filtering the results.
  - After the overhaul, a model based on 1×1 convolutions yields a strong, rotation equivariant generator.

## Equivariance
- 奈奎斯特采样定理，香农抽样定理
- 采样：时域乘以冲击序列，频域卷积冲击序列得到频移
- 图2左，离散域卷积差值滤波器恢复模拟信号，模拟信号点乘冲击序列得到离散信号，考虑矩形窗对sinc函数
- 图2右，非线性激活函数带来高频分量，需要低通滤波器




## Reference
[1] [混叠baidu](https://baike.baidu.com/item/混叠/6996184?fr=aladdin)