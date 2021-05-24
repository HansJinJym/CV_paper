# Attention is All You Need

论文：https://arxiv.org/abs/1706.03762
仓库：[Official TensorFlow](https://github.com/tensorflow/tensor2tensor)，[PyTorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## A High-Level Look
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YMXtV7tadJKic2hdFrgdmYzj5I1FS1tv4tCicygJYd7Ficr0LWnqBNrSjw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
一种新的layer，叫self-attention，它的输入和输出和RNN是一模一样的，输入一个sequence，输出一个sequence，它的每一个输出都看过了整个的输入sequence，这一点与bi-directional RNN相同。但是它的每一个输出都可以并行化计算。
<br>
![](https://jalammar.github.io/images/t/the_transformer_3.png)
![](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)
![](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)
文章中采用6个Encoder/Decoder块堆叠的方式。
<br>
![](https://jalammar.github.io/images/t/Transformer_decoder.png)
一个单词embedding之后传入Self-Attention层，来计算该词与其他词之间的相关性，然后进入前向传播模块。解码器中多了一个Attention模块，使decoder可以更加专注于输入句子中与当前输入的最相关部分。

## Attention中的张量流
![](https://jalammar.github.io/images/t/embeddings.png)
将输入的单词用embedding algorithm转化成512维向量。
![](https://jalammar.github.io/images/t/encoder_with_tensors.png)
![](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)
完整encoder流程如上图，其中Self-Attention层会计算输入之间的相关性，而前向传播中则不计算。
<br>
[Attention可视化](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)，可以看到不同层的attention模块关注的句子位置。

## Self-Attention机制与Q，K，V的理解
### High Level
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YrX5IkL2Jho4BQwHJDBTbQnWK1Dq98L3maJknqkqib09PaKIsxeGxWTw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
**其中Q代表Query，“代表了”当前词汇，用于和其他单词做匹配。K代表Key，用于被其他单词匹配。在对当前单词做运算时，计算出的Q和其他单词的K做运算，得到相关性。在对其他单词运算时，当前单词计算出的K用于和其他单词的Q做运算，得到相关性。即Q是”to match others“，K是”to be matched“。V代表了当前单词提取出的信息，用于后期加权。**
上图中x<sup>i</sup>是一个sequence input vector，embedding得到a<sup>i</sup>，乘上三个不同的转移矩阵W<sub>q</sub>，W<sub>k</sub>，W<sub>v</sub>得到q<sup>i</sup>，k<sup>i</sup>，v<sup>i</sup>。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Yu7XbgXbrH6aic1QpsyfJVcCyYia6s6jcibeBNC6DDucyzWfiaLxMfBfIlw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
接下来使用每个query去对每个key做attention，attention就是匹配这2个向量有多接近。将当前单词的q与其他单词的k做内积，再除以根号q和k的维度（q和k维度相同），得到$\alpha$<sub>i</sub>。因为q·k的数值会随着dimension的增大而增大，所以要除以$\sqrt{dimension}$，相当于归一化的效果。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Y8B375WFtIyTjrAlV4aAUGC7HYLtdgI1WrTE4CvPTM3Nf6ibubeIPaGA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
将$\alpha$<sub>i</sub>取Softmax操作后，得到$\hat{\alpha}$<sub>i</sub>，之后和所有v<sup>i</sup>相乘，结果得到b<sup>i</sup>，即产生b<sup>i</sup>的过程用到了输入的全部信息，并且可以并行计算。如果要考虑local information，则只需要学习出相应的$\hat{\alpha}$<sub>i</sub>=0即可。考虑global information，则只需要学习出相应的$\hat{\alpha}$<sub>i</sub>不为0即可。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YQXMXznHtfW0ialqAn1IBCqfm7MIXnUzuZGCCyRkf8TW8OLw5icQzic7Dg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YibdT4GhVicA5SdzXZfbvkqvSiaahvEbttt9ICfUwuAZDBRRqpqtanSkGQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
其余b<sup>i</sup>的计算过程同理，输入x<sup>i</sup>即可以得到b<sup>i</sup>。最终Attention的High-level look如下图。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YPHOZ9VujENwxHnEqmEcZeItqp6Bx9LfateNnseWFXbclGOvFpZjqvw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### Details in Vectors and Matrices
![](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)
第一步将输入词汇embedding之后，通过W<sup>Q</sup>、W<sup>K</sup>、W<sup>V</sup>三个矩阵得到每个输入单词对应的Q、K、V三个向量。Q、K、V维度比embedding要小，在论文中Q、K、V是64维，而embedding是512维，维度只是一种architecture choice。
![](https://jalammar.github.io/images/t/transformer_self_attention_score.png)
第二步计算每个单词相对当前单词的分数。用当前单词的query（to match others）点乘以每个单词的key（to be matched），即可。
![](https://jalammar.github.io/images/t/self-attention_softmax.png)
第三步将计算出的分数除以8，即$\sqrt{d_k}$（$d_k=64$)，第四步则将计算出来的全部结果取softmax操作，使所有值大于零且相加等于1。此处softmax的结果可以理解为单词之间的相关度，越大则相关度越高。
![](https://jalammar.github.io/images/t/self-attention-output.png)
最后将softmax得到的结果与value（information to be extracted）相乘，可以发现与当前单词越相关的单词，value的权重越大。

将全部输入堆叠成矩阵，流程如下列图所示。
![](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)
![](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
<br>
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Y0A2uAgx4sCEwmbqAP9h4xibIMqBicc896HqWWvOPy3f8TjoxsicQhxqjg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Y0A2uAgx4sCEwmbqAP9h4xibIMqBicc896HqWWvOPy3f8TjoxsicQhxqjg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YHsQ8ankETn3W6zoGdNYYMXbaddGibHicGIc0S4StVlEHtU0tGwEtmiaFA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1Yob0TrzwhFqzBgWhvWEfzZk1H4lJreAfqJ7z3vCicvtHHlly3btqPYcQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YCW8DnfxFYQFzLuFL4Vx92gmGkBQic3FqaicBtibibfxGHRUj6t6jp9Ho3g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YtvG4oKaMLDDPSQffcslksQc4VVp1mibpibXnRG49meDqK9riabhUewJAA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## Multi-head Self-Attention
Multi-head和single-head没有本质区别，只是在计算Q、K、V矩阵时会计算出多个矩阵。下图给出2-head的示例。在最后会有一个转移矩阵$W^O$来将b的维度调整至与原来相同。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YhjUJIOhNHibRRKM7kA4v3qGaselVLQrPbAibZ88icia6IXp3aAvaAfZTfw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YhjUJIOhNHibRRKM7kA4v3qGaselVLQrPbAibZ88icia6IXp3aAvaAfZTfw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YicpxYXJ0FYIAXrUCp3svvLERzeApWbfYsS9XW1RDXkpv54OjFOA3dEA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
下列图也是一组示例。
![](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)
![](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)
![](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)
![](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

## Demo
[Tensor2Tensor Notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)

## Positional Encoding
以上是multi-head self-attention的原理，但是还有一个问题是：现在的self-attention中没有位置的信息，一个单词向量的“近在咫尺”位置的单词向量和“远在天涯”位置的单词向量效果是一样的，没有表示位置的信息(No position information in self attention)。所以输入"A打了B"或者"B打了A"的效果其实是一样的，因为并没有考虑位置的信息。
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YAEovTKOqxU9bOoU9icKvwWYvgnEmfNDLvEc8W5m53shiblX4FaRPBqWg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfqZuR6BRxTDDm1ic4xiaPIJ1YKp86GKC5hXb2vRHzVjvvl2631ChpkmBLibrDGEqKuVrv1bNvzePkQRQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
具体的做法是：给每一个位置规定一个表示位置信息的向量$e_i$，让它与$a_i$加在一起之后作为新的$a_i$参与后面的运算过程，但是这个向量$e_i$是由人工设定的，而不是神经网络学习出来的。每一个位置都有一个不同的$e_i$。
![](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)
位置编码设定类似下图。
![](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

## 完整流程
![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)
![](https://jalammar.github.io/images/t/transformer_decoding_1.gif)
![](https://jalammar.github.io/images/t/transformer_decoding_2.gif)


## Reference
[1] [一篇英文博客](https://jalammar.github.io/illustrated-transformer/)
[2] [Transformer中文综述](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&chksm=ec1c9073db6b1965d69cdd29d40d51b0148121135e0e73030d099f23deb2ff58fa4558507ab8&idx=1&mid=2247531914&scene=21&sn=3b8d0b4d3821c64e9051a4d645467995#wechat_redirect)