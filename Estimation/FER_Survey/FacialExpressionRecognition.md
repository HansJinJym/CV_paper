# Facial Expression Recognition (FER) 面部表情识别

- 其实可以看出，主要还是基于静态图像的FER。主流的方式，从传统的手工特征（LBP,LBP-TOP等），浅层学习（SVM，Adaboost等），深度学习（CNN，DBN，RNN）。从2013年开始，开始有了表情识别的比赛，如FER2013,EmotiW等。
- 整个人脸表情识别的研究是跟随人脸识别的发展而发展的，人脸识别领域比较好的方法会同样适用于表情识别（这里主要是指静态图像FER，毕竟是个分类问题，跟人脸识别类似）。该综述从算法和数据库两方面调研了人脸表情识别领域的进展。在数据库方面，表情识别逐渐从传统的实验室统一控制下的小样本量数据库转移到了现实生活中的多样化大规模数据库。在算法方面，传统的手工设计特征乃至浅层学习特征也不再能很好地适应现实世界中种种与表情无关的干扰因素，例如光照变换，不同的头部姿态以及面部阻挡。于是越来越多的研究开始将深度学习技术运用到了人脸表情识别之中，来解决上述问题。
![](https://img-blog.csdn.net/2018053009195261)

- 完整FER流程图如下
![](https://img-blog.csdn.net/20180530105151962)

- 常用FER数据集
- 数据集主要包含基于静态图像和基于动态序列的数据
- 大部分是7种基础表情，生气，害怕，厌恶，开心，悲伤，惊讶，中立
- 本质是一个分类网络
![](https://img-blog.csdn.net/20180530135227911)

- 基于静态图像的分类方法
    - 预训练模型微调
        - 主要采用分类网络，或者人脸识别网络，相对来说，后者更好。只是有各种fineturn的方式，比如分级、固定某些层，不同层采用不同数据集
        - 还有，考虑到人脸识别模型弱化了人脸情绪差异，可以用人脸识别模型提取特征，然后用表情识别网络消除人脸识别模型带来情绪差异的弱化。也就是这里人脸识别模型起到初始化表情网络的作用
    - 差异化网络输入
        - 实际上就是，除了常见的输入RGB原始脸部数据给网络，还有一些手工特征，如SIFT，LBP，MBP，AGE（3D angle,gradient edge）,NCDV,还有LBP+HOG+Gray@51点landmark生成DSAE，PCA以及裁剪出五官进行特征学习而不是整个脸部等
    - 辅助块或者层改进
        - 基于经典CNN网络架构，一些人设计了更好的网络块或者网络层，如HoloNet(CReLU代替ReLU以及改进的残差块)
    - loss
        - Softmax在表情识别领域，不是很合适，毕竟表情的类间区分本来就不高，这也是难点所在，所以，用人脸识别模型中的改进损失如A-Softmax等效果比普通分类的Softmax普遍好。综述作者也整理了了几种针对表情分类的loss，如基于center loss 改进的ISLand loss（增加类间距离）,LP Loss（locality-preserving,减小类内距离）。基于triplet loss改进的exponential triplet-based loss(网络中增加困难样本的权重)，(N+M)-tupes cluster loss(降低anchor的选择难度，以及阈值化triplet不等式)
    - 网络集成
        - 集成的方式，在机器学习上面非常成功，如Adaboost就是一个很成功的例子，这里讲的是网络结构的集成，要考虑两点，一是网络模型要有充分多样性这样才可以具有互补性，二是一个可靠的集成算法
        - 对于集成算法，这里需要考虑两点，一个是特征集成，另外一个就是输出的决策集成，对于特征集成，最常见的是不同网络模型的特征直接连接
        - 决策集成，如多模型输出加权投票，简单平均，加权平均，甚至可以一起学习每个模型的集成权重
    - 多任务网络
        - 表情和landmark一起、表情和人脸验证一起、以及表情和AUs分类一起。如disBM(高阶玻尔兹曼机)，学习与表情有关的主要坐标以及后续的表情分类，值得注意的有SJMT解决AUs的多标签用以识别AUs，IACNN包含两种提取网络，一路用表情感知测度学习提取判别表情类别的特征，一路用身份感知测度学习提取表情中不变特征，类似的还有MSCNN，基于监督的表情识别和人脸验证一起
    - 网络级联

![](https://img-blog.csdn.net/20180530170435916)



## Reference
- [CSDN survey](https://blog.csdn.net/qq_42393859/article/details/90234622)
- [CSDN notes](https://blog.csdn.net/missyang99/article/details/86542044)