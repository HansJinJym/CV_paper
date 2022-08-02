# Learning Transferable Visual Models From Natural Language Supervision

- CLIP
- OpenAI 2021年发表
- ICML2021
- [official repo](https://github.com/OpenAI/CLIP)

## Overview
- Transferable Visual model 指不使用特定数据集的数据训练模型，但是得到的模型却可以在多个不同的特定数据集上表现出良好的性能，该模型具有Transferable的性质
- From natural language supervision 指从语言文本中提取有效的信息，辅助CV模型的构建和训练
- CLIP是一个预训练模型，就像BERT、GPT、ViT等预训练模型一样。首先使用大量无标签数据训练这些模型，然后训练好的模型就能实现，输入一段文本（或者一张图像），输出文本（图像）的向量表示。CLIP和BERT、GPT、ViT的区别在于，CLIP是多模态的，包含图像处理以及文本处理两个方面内容，而BERT、GPT是单文本模态的，ViT是单图像模态的
- 传统CV有监督训练限制了模型的通用性，经过预训练，自然语言被用来参考学习的视觉概念(或描述新的概念) ，使模型可以zero-shot迁移至下游任务。在NLP中，像GPT-3这种模型，与任务无关的架构能够实现zero-shot到下游数据集，而CV等其他领域，仍然需要在有标注的数据集中进行运算，并且很难迁移至下游，也很难做到zero-shot

## 技术
- 对比学习，只需正样本负样本的定义，Fig.1对角线即位正样本，其余为负样本
- zero shot利用prompt template，得到文本特征（训练集是句子，所以不能仅用单词做zero-shot），图片特征进而计算余弦相似度。摆脱class label，可以预测新的物体，用文本特征摆脱训练集标签限制，实现zero-shot，同时可以完美迁移至下游任务，不需重新训练。同时传统图像算法最终有一个检测头（N选1），也是抑制表现之一
- 应用：StyleCLIP, CLIPDraw, ViLD等
- 大规模数据集，大规模模型

## 方法
- 核心：运用自然语言的监督信号训练视觉模型
- 好处：不需要预先标注数据集，使数据集很容易变大，并且现在只需要图像文本对，限制低，泛化性高；多模态特征学习实现zero-shot迁移
- 爬取四亿对训练图像文本对构建数据集WIT（WebImage Text）
- 图像编码器可以采用ResNet或Vision Transformer，文本采用Transformer

## Reference
- [ref1](https://blog.csdn.net/me_yundou/article/details/123033447)
- [ref2](https://blog.csdn.net/qq_42014059/article/details/122045800)