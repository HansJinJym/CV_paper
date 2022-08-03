# Vision Transformer

- An Image is Worth 16X16 Words: Transformers for Image Recognition at Scale
- ICLR2021
- Google Research
- Vision Transformer将CV和NLP领域知识结合起来，对原始图片进行分块，展平成序列，输入进原始Transformer模型的编码器Encoder部分，最后接入一个全连接层对图片进行分类，在大型数据集上表现超过了当前SOTA模型

## 引言
- 核心问题：把图片这种二维数据转换成一维序列，送进Transformer学习自注意力
- 目的：Transformer在NLP里扩展的很好，越大数据集、越大网络，模型效果会越来越高，没有任何饱和现象，因此CV能否使用
- 由于Transformer输入序列每两个元素之间都要互相做计算，因此是$O(n^2)$级别的复杂度，现在硬件仅支持几百或几千的序列长度，例如BERT采用512长度的输入序列。如果直接把图像转换成所有像素点序列，由于图像像素点过多，因此直接使用不可行，即Transformer不能直接使用至CV中。因此过去一些工作就是CNN+Attention，或者全部使用self-attention
- ViT的处理方法是把图片变成很多个patch，每一个patch大小是16x16，如果输入图像大小为224x224，Transformer输入序列即为14x14=196，大大降低序列长度。然后每一个patch通过一个全连接层，得到linear embedding，送入Transformer，此时一个图像块相当于一个单词
- 采用有监督训练，NLP里基本都采用无监督需训练（如language modeling，masked language modeling），CV更多采用有监督训练方式

## 模型
![](https://img-blog.csdnimg.cn/2021071520481093.png)

- 先把输入图片打成patch，把patch们变成一个序列，每个patch经过线性投射层得到一个特征（patch embedding），加入位置编码，此时每一个token既包含图像信息也包含patch位置信息。
- 类似BERT，加入特殊字符class，位置编码恒为0，由于自注意力会两两之间全部操作，因此0号也会从其他位置学到信息，最后只需将0号的输出做分类判断即可
- 输入图像224x224x3，分patch后包含196个16x16的patch，每一个patch是16x16x3，因此维度变为196x768。线性投射层是一个768x768的全连接层。因此最后Transformer的输入还是196x768，相当于196个维度为768的token。再加上维度为1x768的class token，所以最终整体输入是197x768的token
- 位置信息是一个可学习的向量表，每一个位置向量也是1x768，将位置信息和token做加法
- embedded patches是输入，197x768的tensor。经过layer norm维度不变，经过多头自注意力得出qkv，维度均为197x768/#head（例如ViT-base，12头自注意力，qkv维度为197x64），最后多头进行拼接再次得到197x768。再经过layer norm，维度不变，最后经过MLP，MLP会把维度放大（一般放大四倍），最终会再把维度投射回去，因此最终输出还是197x768。在此基础上可以叠加多个Transformer encoder
- 对于位置编码，不加位置编码也可以但效果较差（即用全部输出序列类似CNN最后一层，展平之后全连接进行分类），1d、2d、相对位置编码都可以，效果差不多。最终为了尽可能和NLP里的transformer结构相似，仍采取1d位置编码
- 最终输出，将0号位置取出即可，即class token输入对应的输出
- 变体包括，base、large、huge（堆叠层数变大，头变多等），ViT-B/16表示base模型采用16x16的patch。注意，patch size越小，训练越难，因为patch小导致token size变大

## Reference
- [ref1](https://blog.csdn.net/weixin_44106928/article/details/110268312)
- [ref2](https://blog.csdn.net/qq_39478403/article/details/118704747)