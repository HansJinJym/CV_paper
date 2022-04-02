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

## Reference
- [ref1](https://blog.csdn.net/me_yundou/article/details/123033447)
- [ref2](https://blog.csdn.net/qq_42014059/article/details/122045800)