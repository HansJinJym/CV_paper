# ICCV2021 Pose Estimation

1. HandFoldingNet: A 3D Hand Pose Estimation Network Using Multiscale-Feature Guided Folding of a 2D Hand Skeleton
   - [paper](https://arxiv.org/abs/2108.05545)
   - [code](https://github.com/cwc1260/HandFold)
   - Folding思路，点云图force至2D骨架关键点，然后fold成3D姿态
   - PointNet++ Encoder, Global Folding Decoder, Local Folding Block

2. Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image
   - [paper](https://www.yangangwang.com/papers/ZHANG-ITH-2021-08.pdf)
   - [code](https://github.com/BaowenZ/Intershape)
   - [project](https://baowenz.github.io/Intershape/)
  ![](https://baowenz.github.io/Intershape/resources/images/fig1_webpage.png)
   - 传统方法都是对单手进行检测或重建，这类方法用到双手任务上会变差。因此本文主要解决双手交叉问题，可以对双手进行重建。
  ![](https://baowenz.github.io/Intershape/resources/images/architecture.png)

3. CPF: Learning a Contact Potential Field to Model the Hand-object Interaction
   - [code](https://github.com/lixiny/CPF)
   - 手物交互
  ![](https://github.com/lixiny/CPF/raw/main/teaser.png)

## Reference
[1] https://github.com/xinghaochen/awesome-hand-pose-estimation#2021-iccv
[2] https://github.com/extreme-assistant/ICCV2021-Paper-Code-Interpretation/blob/master/ICCV2021.md#SGG