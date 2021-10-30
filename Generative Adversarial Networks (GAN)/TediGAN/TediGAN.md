# TediGAN

- visual-linguistic相似模块，确保两个embedding在w空间内足够近
- 文本编辑图像，图像与文本embedding之后，进行style mixing，最后进行实例级优化

## StyleGAN Inversion Module
- 提出两点修改
  - 1. 用真实图像训练，使之未来更能适应真实图像
  - 2. 不在z域，在x域训练，具备更多语义信息

## Visual-Linguistic Similarity
- 目的：视觉和语义信息embed至w空间内

## Instance-Level Optimization
- 目的：保证前后人物一致