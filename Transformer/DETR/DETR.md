# End-to-End Object Detection with Transformers

- [论文](https://arxiv.org/pdf/2005.12872.pdf)
- [代码](https://github.com/facebookresearch/detr)
- 作者：Facebook AI
- DEtection TRansformer
  1、提出了一种目标检测新思路，真正的end-to-end，更少的先验（没有anchor、nms等）
  2、在coco上，准确率、运行效率与高度优化的faster R-CNN基本持平。在大目标上效果比faster R-CNN好
  3、与大多数现有的检测方法不同，DETR不需要任何自定义层，因此复现容易，涉及到模块都能在任何深度学习框架中找到
  4、不需要人工设计的模块组件，如非极大抑制、anchor生成等
  5、将目标检测问题当做一个直接集合预测问题（direct set prediction problem）
  6、本文的任务是Object detection，用到的工具是Transformers，特点是End-to-end。


目标检测的任务是要去预测一系列的Bounding Box的坐标以及Label， 现代大多数检测器通过定义一些proposal，anchor或者windows，把问题构建成为一个分类和回归问题来间接地完成这个任务。**文章所做的工作，就是将transformer运用到了object detection领域，取代了现在的模型需要手工设计的工作，并且取得了不错的结果**。在object detection上DETR准确率和运行时间上和Faster RCNN相当；将模型 generalize 到 panoptic segmentation 任务上，DETR表现甚至还超过了其他的baseline。DETR第一个使用End to End的方式解决检测问题，解决的方法是把检测问题视作是一个set prediction problem，网络基本结构如下图。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Ybz8M92N5uxL20t9VfIqjlsMExuduAMe8GvxmaWrxb62mnRsxc81iaAA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
DETR结合CNN和Transformer的结构，并行实现预测。网络的主要组成是CNN和Transformer，Transformer借助self-attention机制，可以显式地对一个序列中的所有elements两两之间的interactions进行建模，使得这类transformer的结构非常适合带约束的set prediction的问题。DETR的特点是：一次预测，端到端训练，set loss function和二分匹配。


文章的主要有两个关键的部分。
第一个是用transformer的encoder-decoder架构一次性生成$N$个box prediction。其中$N$是一个事先设定的、远远大于image中object个数的一个整数。
第二个是设计了bipartite matching loss，基于预测的boxex和ground truth boxes的二分图匹配计算loss的大小，从而使得预测的box的位置和类别更接近于ground truth。
DETR整体结构可以分为四个部分：backbone，encoder，decoder和FFN。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YqstusedM0UHutdFibXj6QxnLFeIFnaJY7dSwO1Ficc7a3DwgbFWnqROw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
1. backbone先将输入图像转为feature map
2. encoder先将feature map用1x1卷积把通道数压缩，然后将空间的维度（高和宽）压缩为一个维度，再引入位置编码
   ![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YFzEl1RibOG8HicwQMULq9tsz6q6fvk8l4ZNuS0PftV7FYYgWqLJy7g0g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
   参考[1]，有待进一步理解。相加前，feature map和位置编码维度都是（B，d=256，H，W），然后通过序列化变成（HxW，B，256），相加后输入至encoder。
   ![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Y5RvFicyaSKKQvFNxpA2p91SekSvtjsBZKxvyleV550mCdNicH1uB9ibvw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
   原版Transformer只在Encoder之前使用了Positional Encoding，而且是在输入上进行Positional Encoding，再把输入经过transformation matrix变为Query，Key和Value这几个张量。但是DETR在Encoder的每一个Multi-head Self-attention之前都使用了Positional Encoding，且只对Query和Key使用了Positional Encoding，即：只把维度为（HxW，B，256）维的位置编码与维度为（HxW，B，256）维的Query和Key相加，而不与Value相加。详细流程如下图
   ![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YNxFd79h5rRMHic9rIeE4BMGoXoA6JYkyxb3wboOuKd2iaZzcSTUnJ0ibw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
   3. DETR的Decoder和原版Transformer的decoder是不太一样的。原版Transformer，decoder的最后一个框：output probability，代表我们一次只产生一个单词的softmax，根据这个softmax得到这个单词的预测结果。这个过程我们表达为：predicts the output sequence one element at a time。不同的是，DETR的Transformer Decoder是一次性处理全部的object queries，即一次性输出全部的predictions；而不像原始的Transformer是auto-regressive的，从左到右一个词一个词地输出。这个过程我们表达为：decodes the N objects in parallel at each decoder layer。
   Object queries是一个维度为(100, B, 256)维的张量，数值类型是nn.Embedding，说明这个张量是可以学习的，即：我们的Object queries是可学习的。Object queries矩阵内部通过学习建模了100个物体之间的全局关系，例如房间里面的桌子旁边(A类)一般是放椅子(B类)，而不会是放一头大象(C类)，那么在推理时候就可以利用该全局注意力更好的进行解码预测输出。
   到了每个Decoder的第2个multi-head self-attention，它的Key和Value来自Encoder的输出张量，维度为(HW, b, 256)，其中Key值还进行位置编码。Query值一部分来自第1个Add and Norm的输出，维度为(100, b, 256)的张量，另一部分来自Object queries，充当可学习的位置编码。所以，第2个multi-head self-attention的Key和Value的维度为(hw, b, 256)，而Query的维度为(100, b, 256)。
   Object queries充当的其实是位置编码的作用，只不过它是可以学习的位置编码，所以，我们对Encoder和Decoder的每个self-attention的Query和Key的位置编码做个归纳.
   ![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Y0nDGCo9GDdU15NsUygG0U5sHSfBThbqjCVMOjsXibC9Fd0bXBXGWgpg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

   ### Reference
   [1] [Transformer综述一](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&chksm=ec1c9073db6b1965d69cdd29d40d51b0148121135e0e73030d099f23deb2ff58fa4558507ab8&idx=1&mid=2247531914&scene=21&sn=3b8d0b4d3821c64e9051a4d645467995#wechat_redirect)
