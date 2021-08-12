# Object Detection

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfpoNPRibFklKkeenx5mob3viace7l9TSt14Y1n4QmsJ3ibynsIXq1FnlSZxUyiaDfyYyEmFrh20oANADw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 基于阶段数不同
- 两阶段目标检测算法因需要进行两阶段的处理：1）候选区域的获取，2）候选区域分类和回归，也称为基于区域（Region-based）的方法。与单阶段目标检测算法的区别：通过联合解码同时获取候选区域、类别。
- 【两阶段】和【多阶段】目标检测算法统称级联目标检测算法，【多阶段】目标检测算法通过多次重复进行步骤：1）候选区域的获取，2）候选区域分类和回归，反复修正候选区域。

## 基于是否使用锚框
主要考虑问题：
1、准确性
2、实时性
3、多尺度
4、标签方案
5、目标重叠
6、模型训练
7、重复编码
8、数据增强
9、样本不平衡

## Reference
[1] https://mp.weixin.qq.com/s/ZpE7a6xrG8eqBEnEGbI-fg