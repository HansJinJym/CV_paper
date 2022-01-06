# ResNeSt: Split-Attention Networks

- 提出了一个模块化的Split-Attention block，该block可实现跨feature map groups的attention。 通过以ResNet样式堆叠这些Split-Attention块，获得了一个称为ResNeSt的新ResNet变体。 网络保留了完整的ResNet结构，可直接用于下游任务，而不会引起额外的计算成本
- 由于ResNet模型最初是为图像分类而设计的，由于感受野大小有限且缺乏跨通道交互，它们可能不适合各种下游应用。这意味着要提高给定计算机视觉任务的性能，需要进行“网络手术”来修改ResNet，以使其对特定任务更加有效。
- 创建具有通用改进特征表示的通用backbone，从而提高性能跨通道信息，同时可以很好的迁移至下游任务
- 我们探索了ResNet的简单体系结构修改，将feature map 拆分attention纳入各个网络模块中。 更具体地说，我们的每个block都将特征图分为几组（沿通道维数）和更细粒度的子组或splits，其中，每个组的特征表示是通过其分割表示的加权组合确定的（ 根据全局上下文信息选择权重）。 我们将结果单元称为Split-Attention block，它保持简单且模块化。 通过堆叠几个Split-Attention block，我们创建了一个类似ResNet的网络，称为ResNeSt（代表“ split”）。我们的体系结构不需要比现有ResNet变量更多的计算，并且很容易被用作其他视觉任务的backbone。
- 图像分类和迁移学习应用的大规模基准测试。我们发现，利用ResNeSt backbone的模型能够在几个任务上达到最先进的性能，即：图像分类，目标检测，实例分割和语义分割。


- 多路径表示和split-attention单元

![](https://img-blog.csdnimg.cn/20200526194029230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY3MTQyNQ==,size_16,color_FFFFFF,t_70)

- split-attention模块

![](https://img-blog.csdnimg.cn/20200526194913277.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY3MTQyNQ==,size_16,color_FFFFFF,t_70)


## Reference
- https://blog.csdn.net/weixin_40671425/article/details/106362944