# Overfitting and Underfitting

## 过拟合和欠拟合

![](https://img-blog.csdn.net/20171102211431879?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTgyNTQzODU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 常用的判断方法是从训练集中随机选一部分作为一个验证集，采用K折交叉验证的方式，用训练集训练的同时在验证集上测试算法效果。在缺少有效预防欠拟合和过拟合措施的情况下，随着模型拟合能力的增强，错误率在训练集上逐渐减小，而在验证集上先减小后增大；当两者的误差率都较大时，处于欠拟合状态(high bias, low variance)；当验证集误差率达到最低点时，说明拟合效果最好，由最低点增大时，处与过拟合状态(high variance, low bias)。

![](https://img-blog.csdn.net/20180210231144566?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbmluaV9jb2RlZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

- 防止过拟合的方法（本质：数据太少、模型太复杂）
    - 扩充样本数据，使模型能学到更多类型的数据。可以从数据源头多获取数据，或采用数据增强等方式。
    - 减少网络层数、神经元数量
    - 提前截止训练，early stopping
    - 限制权值（weight-decay），也叫正则化（regularization）
    ![](https://img-blog.csdn.net/20171102210510340?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTgyNTQzODU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
        - 既能降低降低实际输出与样本之间的误差，也能降低权值大小，其中 $\lambda$ 过大会导致欠拟合，过小会导致过拟合
        - 加入正则项，可以使权值普遍变小，相当于弱化每一个神经元，达到防止过拟合的目的
    - dropout，让一部分神经元概率不参与计算
    ![](https://img-blog.csdn.net/20171102211233924?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTgyNTQzODU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
        - 训练过程
            - 以某概率随机地删除一部分神经元
            - 用剩下的神经元前向传播得到误差，采用反向传播对这些保留的神经元进行参数更新，被删除的神经元不更新
            - 恢复被删除的神经元
            - 继续1，2，3至训练结束
        - 测试过程
            - 训练的时候由于以概率1-p随机的丢掉了部分神经元（p的概率保留神经元），测试时用的是全部的神经元，因此输出结果需要乘以p
    - 给网络的输入、权值或响应上增加噪声
    - 贝叶斯方法
    ![](https://pic3.zhimg.com/80/v2-88170130d8bac2f6f54998473ec99b95_1440w.jpg?source=1940ef5c)

![](https://pic3.zhimg.com/80/v2-1c0588c97d1302b0e7bc8c6d5eede473_1440w.jpg?source=1940ef5c)

## 正则化 Regularization
- 机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，一般英文称作 l1-norm 和 l2-norm，中文称作 L1正则化 和 L2正则化，或者 L1范数 和 L2范数。L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓惩罚是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0MjI4MTU4?x-oss-process=image/format,png#pic_center)

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0MzE0MzMz?x-oss-process=image/format,png#pic_center)

- L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择；L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合
- 稀疏矩阵指的是很多元素为0，只有少数元素是非零值的矩阵，即得到的线性回归模型的大部分系数都是0。通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个稀疏模型，表示只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，或者贡献微小（因为它们前面的系数是0或者是很小的值，即使去掉对模型也没有什么影响），此时我们就可以只关注系数是非零值的特征。这就是稀疏模型与特征选择的关系。
- L2不具有稀疏性，因为不太可能出现大部分权值为0的现象。[link4]
- 将正则项看作拉格朗日乘子，不带正则项的原损失画出其等高线，相交的地方则为最优解，本质上是一个带约束的优化问题。考虑正则化的范数，L1是方形，在各个坐标轴上是尖点，容易在此处相交，坐标轴上相交时相当于有其他参数是零，达到产生稀疏矩阵的效果；L2是圆形，因此各点等概率相交，最终表现为将各个参数变小，达到预防过拟合的效果；范数越高，对应曲线越接近标准正方形。

## 归一化 Normalization
- 五种归一化，Batch Norm，Layer Norm，Instance Norm，Group Norm，Switchable Norm，论文见[link6]
- 神经网络学习过程的本质就是为了学习数据分布，如果我们没有做归一化处理，那么每一批次训练数据的分布不一样，从大的方向上看，神经网络则需要在这多个分布中找到平衡点，从小的方向上看，由于每层网络输入数据分布在不断变化，这也会导致每层网络在找平衡点，显然，神经网络就很难收敛了。当然，如果我们只是对输入的数据进行归一化处理（比如将输入的图像除以255，将其归到0到1之间），只能保证输入层数据分布是一样的，并不能保证每层网络输入数据分布是一样的，所以也需要在神经网络的中间层加入归一化处理。
- BN、LN、IN和GN这四个归一化的计算流程几乎是一样的，可以分为四步：
    - 1.计算出均值
    - 2.计算出方差
    - 3.归一化处理到均值为0，方差为1
    - 4.变化重构，恢复出这一层网络所要学到的分布

![](https://img-blog.csdnimg.cn/20190817111941649.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTMyODkyNTQ=,size_16,color_FFFFFF,t_70)

- 训练的时候，是根据输入的每一批数据来计算均值和方差。测试时，对于均值来说直接计算所有训练时batch均值的平均值，然后对于标准偏差采用每个batch方差的无偏估计，即全部训练数据的均值方差。
    - 训练时，如果用全部训练数据的均值方差，容易产生过拟合。对于BN，其实就是对每一批数据进行归一化到一个相同的分布，而每一批数据的均值和方差会有一定的差别，而不是用固定的值，这个差别实际上能够增加模型的鲁棒性，也会在一定程度上减少过拟合。也正是因此，BN一般要求将训练集完全打乱，并用一个较大的batch值，否则，一个batch的数据无法较好得代表训练集的分布，会影响模型训练的效果。

![](https://img-blog.csdnimg.cn/20190817112054946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTMyODkyNTQ=,size_16,color_FFFFFF,t_70)

- Batch Norm
    - BN的计算就是把每个通道的NHW单独拿出来归一化处理
    - 针对每个channel我们都有一组γ,β，所以可学习的参数为2*C
    - 当batch size越小，BN的表现效果也越不好，因为计算过程中所得到的均值和方差不能代表全局
- Layer Norm
    - LN的计算就是把每个CHW单独拿出来归一化处理，不受batch size的影响
    - 常用在RNN网络，但如果输入的特征区别很大，那么就不建议使用它做归一化处理
- Instance Norm
    - IN的计算就是把每个HW单独拿出来归一化处理，不受通道和batch size的影响
    - 常用在风格化迁移，但如果特征图可以用到通道之间的相关性，那么就不建议使用它做归一化处理
- Group Norm
    - GN的计算就是把先把通道C分成G组，然后把每个gHW单独拿出来归一化处理，最后把G组归一化之后的数据合并成CHW
    - GN介于LN和IN之间，当然可以说LN和IN就是GN的特例，比如G的大小为1或者为C
- Switchable Norm
    - 将 BN、LN、IN 结合，赋予权重，让网络自己去学习归一化层应该使用什么方法
    - 集万千宠爱于一身，但训练复杂

## Reference
- [link1](https://blog.csdn.net/weixin_42575020/article/details/82949285)
- [link2](https://zhuanlan.zhihu.com/p/462183031)
- [link3](https://www.zhihu.com/question/20924039)
- [link4](https://blog.csdn.net/jinping_shi/article/details/52433975)
- [link5](https://blog.csdn.net/u013289254/article/details/99690730)