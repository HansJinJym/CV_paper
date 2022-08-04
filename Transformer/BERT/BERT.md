# BERT

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- 在NLP任务中，首次可以用一个大的数据集训练好一个深的神经网络，然后应用在很多其他NLP任务上，简化了NLP任务的训练，又提升了性能
- 语言理解任务的深度双向Transformer，GPT是一个单向模型，现在预测未来，而BERT双向在于过去和未来同时包括。ELMo是双向模型，不过用的是较老的双向RNN。本文在二者之上构建双向Transformer架构的BERT

## 介绍
- 单向模型的局限性：对于QA这种token-level的任务，希望看完整个句子来选答案，因此需要全局信息，即双向信息
- BERT解除单向的限制，用到的是MLM，masked language model，相当于完型填空，每次随机选一些token给mask住，然后目标函数是预测被mask的token。MLM相当于允许使用左侧和右侧信息，成为双向模型
- 此外，还使用next sentence prediction，即给出两个句子来预测原文中是否是相邻的，以此学习句子层面的信息

## 网络
- 无监督预训练 + 有监督微调
- 多层双向Transformer编码器
- 附：模型参数量计算（参考李沐BERT-bilibili），嵌入层输入为字典大小30000，输出为隐藏层数量H（768or1024），自注意力qkv加上输出层的投影矩阵全部为HxH（与多头注意力的头数无关，因为多头最后会拼接回去），MLP包含两层，第一层H到4H，第二层4H到H，因此总参数量为 $30000H + 12H^2L$，其中L为transformer encoder块堆叠的个数
- 输入序列第一个token永远是特殊token [CLS]，输出希望是整个序列的信息，自注意力两两之间都会参与计算，因此位置影响不大。输入两个句子（因为原始Transformer编码器解码器都会输入一个句子，BERT只有编码器，所以需要放一起），两个句子用特殊token [SEP] 分开
- 输入有三部分组成，第一个是输入序列的token embeddings，第二个是segment embeddings，即区别是第一句话还是第二句话，第三个是position embedding，即token的位置编码，然后全部叠加。Transformer里的位置编码是人为固定的，BERT是可学习的
- 预训练MLM。token有15%的概率被替换为mask（两个特殊token除外），但是由于微调不存在MLM，导致预训练和微调数据有差异。最后作者采用15%的概率选中这个token，选中后80%概率替换为特殊token [MASK]，10%的概率替换为随机其他token，10%概率不操作（这是为了模拟后面二阶段微调时的数据）
- 预训练NSP。50%概率两个句子在原文中相邻，50%概率随机取不相邻的。这种方法对QA和语言推理任务上效果提升显著
- 下游任务微调。BERT的结构可以更关注上下文，但是不能做机器翻译任务，因为机器翻译任务的普通transformer结构的编解码器看不到对方的句子，BERT的自注意力可以同时看到两个句子。另一个好处在于，下游任务只需要针对特定任务修改输入输出层，不需要修改整体结构