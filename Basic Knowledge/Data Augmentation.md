# 数据增强

## Mixup, Cutout, CutMix
- Mixup: 将随机的两张样本按比例混合，分类的结果按比例分配
- Cutout: 随机的将样本中的部分区域cut掉，并且填充0像素值，分类的结果不变
- CutMix: 就是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配
  
  
  ![](https://img-blog.csdnimg.cn/20200116092100859.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODcxNTkwMw==,size_16,color_FFFFFF,t_70)


- CutMix也可应用于小样本数据集
- 仓库地址： https://github.com/clovaai/CutMix-PyTorch