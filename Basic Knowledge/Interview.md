# CV Interview

## 网络模型
- 最常见的是ResNet的网络结构，如何解决网络退化问题
- 网络的参数量计算量如何计算；简单的量化原理
- 轻量级网络；以及一些比较新的网络要知道，ResNeSt等

## 检测
- 检测作为大部分工程的基础方向，还是要精通
- 常被问到的有faster rcnn的anchor和gt匹配问题
- 两阶段与一阶段的优缺点
- Anchor Free和Anchor Based的区别联系，各自经典方法
- IOU的计算代码，从两个框，三个框，到多个框（涉及到Python的broadcast机制）
- NMS及其变种

## Transformer
- QKV的理解
- 为什么要缩放
- 绝对位置编码和相对位置编码
- Transformer和MLP，CNN的区别联系
- ViT，Swin等经典工作

## BN
- BN的原理和代码
- 与LN的区别联系
- Sync BN的方差计算

## 分割
- 经典方法和最新工作
- 细长物体和小目标的分割
- 常用的loss

## Pytorch
- DDP
- model.train 和 model.val
- dataloader原理

## Python
- 垃圾回收
- GIL
- 深拷贝浅拷贝
- 进程线程

## C++
- 构造函数；虚函数；纯虚函数；智能指针；容器；继承；析构函数；宏与常量；

## 其他
- L1，L2的区别联系，概率密度函数
- KL散度以及为什么大于0
- 最大池化和平均池化的前向反向
- Dropout原理
- 过拟合
- 样本不平衡和噪声问题怎么处理
- 常见的激活函数和初始化方法
- softmax 如何防止溢出

## 前沿动向
- 自监督
- 多模态
- Transformer等的发展现状

## 算法题
- 排序算法；链表相关；二分查找；简单动态规划；DFS，BFS，树很少遇到；

## 开放性问题
- 针对业务场景的设计；


## Reference
[1] [极市面经](https://mp.weixin.qq.com/s/BYPufwGIpzw5pW3Ro0HvOw)