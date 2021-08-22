# GFPGAN

- Towards Real-World Blind Face Restoration with Generative Facial Prior
- 利用生成式人脸先验进行真实世界盲目的人脸重构
- CVPR2021
- 解决的问题： 如何从低分辨率低质量的真实图像中获得较好的先验知识，复原人脸图像

## Introduction
- 盲目的人脸修复的目标是从存在未知退化的低质量人脸中修复出高质量的人脸，如低分辨率、噪声、模糊、压缩伪影等。当应用到现实世界的场景中，由于更复杂的退化，不同的姿态和表情，它变得更具挑战性。以往的研究典型地利用了人脸修复中的特定人脸先验，如人脸关键点检测、解析映射、人脸成分热图，并表明这些几何人脸先验对于修复准确的人脸形状和细节至关重要。然而，这些先验信息通常是从输入图像中估计出来的，并且不可避免地会在输入质量很低的情况下退化。
- 最近，对于人脸复原问题，需要从低分辨率人脸图像中提取几何人脸先验，用于复原人脸。但是许多时候，从现实低分辨图像中所提取的许多人脸的先验都不太准确，并且纹理信息也受限。
- 另一个策略是引入参考先验，即高质量引导人脸或者人脸成分字典，生成真实的人脸结果。但是，高分辨率参考文件的不可访问性限制了它的实际适用性，而字典的固定容量限制了它的面部细节的多样性和丰富性
- 文章提出了一种生成式的人脸先验（GFP），用于真实世界的盲人脸图像复原。由于预训练的人脸生成对抗网络（如styleGAN）所生成的假脸，具有高分辨率，丰富的几何形状、人脸纹理和颜色，这些使到其能够联合用于复原人脸的细节和增强颜色。
- 提出了一种GFP-GAN，其包含了一个退化去除模块和一个预训练的人脸GAN作为人脸先验。通过直接的潜在代码映射和几个通道分割空间特征变换（CS-SFT）层以粗调方式连接。
- contribution
	- 利用丰富多样的生成人脸先验来进行绑定人像复原。这些先验包含足够的面部纹理和生动的色彩信息，能够联合执行人脸修复和色彩增强。
	- 提出GFP-GAN框架，该框架具有精巧的架构设计，并能融合人脸生成先验。带有CS-SFT层的GFP-GAN在一次正向传递中实现了保真度和纹理忠实度之间的良好平衡。
	- 大量实验表明，GFP-GAN在合成数据集和真实数据集上均取得了优于现有技术的性能。

## Methodology
- UNet + StyleGAN prior
- CS-SFT
	-  Channel-Split Spatial Feature Transform
	-  提取输入图片最接近的latent code
	-  提取多分辨率空间特征，类似StyleGAN生成图片的逐级特征
	-  F-GAN和F-Spacial两个输入
- Degradation Remove Module
	- 图像包含有许多不同的退化因素。本文所提出的退化去除模块，用于提取清晰特征；本模块基于U-net网络模型，并提高大范围模糊的适应性和生成不同分辨率的特征。利用金字塔复原指导中间结果。
	- Decoder逐级生成对应图片，与ground truth对比，计算修复loss
- Generative Facial Prior
	- F-latent: Z空间的code
	- F-spatial: F-latent逐级放大每层的结果
	- W: W空间code
	- F-GAN: W生成的结果
- CS-SFT(Channel-Split Spatial Feature Transform)
	- SFT
	![](https://img-blog.csdnimg.cn/2021041022303966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODQ2NjA2,size_16,color_FFFFFF,t_70)
	- CS-SFT: 额外拆出一个通道。输入拆成两部分，identity确保realness，SFT确保fidelity
- Model Objectives
	- Reconstruction Loss: L1 loss + perceptual loss(通过VGG等提取特征，计算特征对应的loss)
	- Adversarial Loss: 与StyleGAN2判别器损失类似
	- Facial Component Loss：首先用ROI对齐来裁剪感兴趣的区域，对于每个区域，训练独立的和小的局部识别器来区分恢复的patches是否真实，将patches推向接近自然的面部成分分布。通过ROI align之后，对每一个ROI部分进行判定；首先，是判定对应的ROI是否真，接着，利用Gram matrix statistics，判断ROI的风格loss
	- Identity Preserving Loss：为了保持人脸 identity 的一致, 使用了人脸 identity 一致损失函数（预训练的ArcFace），即在人脸识别模型的特征空间中去拉近，确保重构结果与GT接近
	- 最终loss为所有loss的叠加
- 实验
	- 和之前大部分工作类似，GFP-GAN 采用了 Synthetic 数据的训练方式。研究者们发现在合理范围的 Synthetic 数据上训练, 能够涵盖大部分的实际中的人脸。GFP-GAN 的训练采用了经典的降质模型, 即先高斯模糊, 再降采样, 然后加白高斯噪声, 最后使用 JPEG 压缩。
	![](https://img-blog.csdnimg.cn/2021041215470573.png)
- 总结
	- 提出GFP-GAN框架，利用丰富和多样化的生成式人脸先验来完成具有挑战性的盲目的人脸修复任务。这一先验被纳入到新的通道分割空间特征转换层的修复过程中，使模型能够很好地平衡真实性和保真度。此外，介绍了人脸组成损失、身份保留损失和金字塔修复指导等精细设计。广泛的比较表明，GFP-GAN在联合面部恢复和真实世界图像的颜色增强方面具有优越的能力，优于现有技术。

- Reference
1. https://blog.csdn.net/qq_38846606/article/details/115560180

