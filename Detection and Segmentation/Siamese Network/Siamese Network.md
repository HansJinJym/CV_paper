# Siamese Network

## Basic Siamese Network
- 共享权值，双分支

![](https://pic2.zhimg.com/80/v2-53093b7e19dea78a36a00fc3a812f921_1440w.jpg)

## SiamFC
- ECCV2016
- 单目标跟踪（SOT）算法，端到端网络，实时，可跟踪首帧框出的任何物体

![](https://pic2.zhimg.com/80/v2-18eff45f279181dad0920e2dc6960689_1440w.jpg)

- 在单目标跟踪任务中，作为基准的模板则是我们要跟踪的对象，通常选取的是视频序列第一帧中的目标对象，而候选样本则是之后每一帧中的图像搜索区域，而孪生网络要做的就是找到之后每一帧中与第一帧中的模板最相似的候选区域，即为这一帧中的目标
- 该结构首先z为输入的范本，即第一帧图像中的目标框，大小为127x127x3，x为输入的搜索图像，大小为255x255x3，接着对两个输入分别进行 $\phi$ 变换（作者采用了AlexNet的网络结构），即为特征提取，分别生成了6x6x128和22x22x128的特征图（feature map），提取了特征之后，再对提取的特征进行互相关操作（即求卷积），生成响应图（heatmap）
- 最后一步卷积操作为，$f(z,x) = \phi(z) * \phi(x) + b$ ，通过卷积运算提取x中与z最为相近的部分，$f$ 即比较z和x相似性的函数
- 响应图中响应值最高的位置就对应着z可能的位置，其中红蓝是两个位置。网络最终生成的是17x17的heatmap，而输入时255x255的搜索区域，为了实现映射关系，作者将17x17的响应图进行双三次插值（16倍）生成272x272的图像来确定物体的位置（事实应该和候选帧尺寸相同）
- 输入resize至定值，不够放缩时采用均值填充，同时由于后序视频帧追踪物体尺度会有变化，采用padding
- 响应图最大值即为原图对应目标中心点，利用多尺度信息（将输入resize变为多尺度形成mini-batch）得到最佳尺度，保持整体目标框长宽比例不变，乘以最佳尺度
- 只能估计目标的中心位置，而要想对目标的尺寸进行估计，只有通过多尺度测试来预测尺度的变化，这种方式不仅计算量冗余，同时也不够精确
- 响应图中响应最大的点即对应原图最相似的一部分，相当于FC的感受野

## SiamRPN
- CVPR2018
- SiamFC的方法只能估计目标的中心位置，而要想对目标的尺寸进行估计，只有通过多尺度测试来预测尺度的变化，这种方式不仅增加了计算量，同时也不够精确。本文引入RPN对bbox回归，提高跟踪定位的精度。

![](https://pic1.zhimg.com/v2-ccfad1783a6fb2f3baafc51c472181e5_1440w.jpg?source=172ae18b)

- 该模型是通过对两个分支的correlation feature map 做相关的proposal抽取。在跟踪任务中，由于没有预先定义的类别，因此作者通过Siameses模板分支将目标的外观信息编码到RPN特征映射中，以区分前景和背景，本质在于待追踪物体没有标签信息并且首帧框中同时包含前景和后景。
- Contributions
    - 作者首次将端到端的离线训练方式，应用在大尺度的图像跟踪任务上
    - 在线跟踪过程中，提出了一种局部单点检测的方法，可以有效地改善传统的多尺度检测方法
- 网络整体分为两部分
    - Siames Network，与SiamFC类似，孪生网络分为上下两支，上下两支路的网络结构和参数完全相同，该网络的作用是分别提取模板帧和检测帧的图像特征。孪生网络的两个分支可以用一个卷积网络实现，值得注意的是这个卷积网络必须为全卷积网络（不能有padding操作），满足平移不变性，即先对图像进行有比例因子的转换操作再进行全卷积操作等同于先对图像进行全卷积操作再进行转换操作，原因在于最后需要通过预测结果通过插值等手段反推原图中的信息。此处卷积采用修改AlexNet层，移除了conv2-conv4
    - Region Proposal Network，该子网络的作用是对bbox进行回归，得到精确的位置估计。RPN网络由两部分组成，一部分是分类分支，用于区分目标和背景，另一部分是回归分支，它将候选区域进行微调。
- 网络具体参数细节
    - 孪生网络：模版分支与检测分支与SiamFC类似，固定输入尺寸，经过特征提取网络变为相应尺度的特征图
    - RPN
        - 分类分支，通过3x3卷积，由于该处特征图每个点会产生k个anchor，每个又分为对应判断前景和背景，因此通道数扩大2k倍
        - 回归分支，通过3x3卷积，同样特征图每个点产生k个anchor，并回归出anchor四个偏移量，因此通道数扩大4k倍

## SiamMask
- CVPR2019
- [demo](https://zhuanlan.zhihu.com/p/76460186?from_voters_page=true) [6]
- SiamFC通过预测候选区域的score map来得到物体的位置，物体的尺度大小通常是通过图像金字塔得到，计算复杂，同时无法得到物体的长宽比变化。
- SiamRPN在物体发生旋转的时候，简单的box的表述通常会产生极大的损失，这实际上就是表述本身存在的缺陷
- 利用VOS解决VOT，one-stage端到端，不需要首帧mask只需要首帧框。提出了对视觉目标跟踪（VOT）和视频目标分割（VOS）的统一框架SiamMask，将初始化简化为视频跟踪的box输入即可，同时得到box和mask两个输出
- [contrast demo](https://vdn.vzuu.com/SD/a0cf4a00-ec53-11ea-acfd-5ab503a75443.mp4?disable_local_cache=1&auth_key=1645611433-0-0-1e893658be1be7f80b3cdeed475e5472&f=mp4&bu=pico&expiration=1645611433&v=ali)

![](https://pic4.zhimg.com/80/v2-bb92bdcdece1bb7c47245c3228688d23_1440w.jpg)

- 网络结构
    - 孪生backbone采用ResNet50的前四个conv，其中模板输入尺寸固定，搜索输入不必固定，得到的特征图做互相关，得到17x17x256的特征图，后面可接三分支结构或两分支结构
    - 前半部分的孪生特征提取网络类似SiamFC，输出的响应图中每一个空间元素即为RoW，response of a candidate window，同时互相关操作改为深度互相关，即通道是256而不是SiamFC中的1，目的是可以编码更多目标物体的信息
    - mask分支对于每一个RoW，通过两个1x1卷积层预测一个mask，图中示例第一层256通道，第二层63x63通道，目的是mask像素分类时可以更精确且避免误检，最后利用由上采样和跳跃连接组成的精修模块融合多尺度特征，得到更加准确的mask
    - 两分支结构中，mask支路相同，score支路维度为17x17x1，通道时为1而不是2k，作用是判定响应图中每一个RoW的置信度，里面根据score挑出一根class分数（score）最高的柱子（RoW）用于生成mask。同理，三分支里面选哪根柱子，可以通过class score分支的最高得分选取。

    ![](https://pic1.zhimg.com/80/v2-a41c0b17e663082ca974293e81a24704_1440w.jpg)

    - 生成mask的两种方式，base path和refine path
        - base path将1x1x63x63resize至63x63，做sigmoid用于判断这个矩阵的某一个值是否属于mask，再用仿射变换映射回原图（cv2.warpAffine()），同时可以借助cv2.contour()和cv2.minAreaRect()生成最小外接矩形
        - refine path先通过score分支找出选取的1x1x256的RoW（这个RoW相当于涵盖mask的全部信息），先做反卷积至15x15x32（应该可以理解为深度信息换宽高信息），然后叠加多尺度特征图并上采样至原始大小。backbone里面多次的pooling，会导致最终分割的精度损失，因为丢失了很多空间信息，所以refine path的想法是想要结合低层的空间信息特征逆转pooling造成的损失，一步步提高mask的resolution，进而提高分割的精度

# DaSiamRPN
- ECCV2018
- Distractor-aware，应对干扰物问题
- VOT的问题：1.常见的siam类跟踪方法只能区分目标和无语义信息的背景，当有语义的物体是背景时，也就是有干扰物（distractor）时，表现不是很好。2.大部分siam类跟踪器在跟踪阶段不能更新模型，训练好的模型对不同特定目标都是一样的。这样带来了高速度，也相应牺牲了精度。3.在长时跟踪的应用上，siam类跟踪器不能很好的应对全遮挡、目标出画面等挑战。
- 主要创新点在于针对性扩充训练数据做数据增强，提出干扰物感知模块，提出局部到全局的搜索策略来重新检测目标以应对跟丢现象

## SiamRPN++
- CVPR2019
- 之前的网络需要满足平移不变性（目标在检测帧中移动时，跟踪的响应位置也会相应地进行等量移动），因此只能采用AlexNet结构，并且直接使用预训练好的深层网络反而会导致跟踪算法精度的下降。因此本文主要解决的问题是将深层基准网络ResNet、Inception等网络应用到基于孪生网络的跟踪网络中
- 破坏平移不变性会导致网络学习到位置偏见，即图像中心处响应会越来越大，但是语义、实例分割不会出现这种问题，原因是分割任务中每个像素中整体分布均匀，而跟踪任务数据绝大部分都出现在图像或视频中央，论文中实验验证了这个结论

![](https://img-blog.csdnimg.cn/20190315161358346.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1daWjE4MTkxMTcxNjYx,size_16,color_FFFFFF,t_70#pic_center)

- 采用ResNet50的骨干，更换采样策略，增加空洞卷积操作，多层特征融合，并引用多层SiamRPN

## SiamFC++
- AAAI2020
- 多分支结构，引入分类和目标状态估计分支（G1），无歧义的分类得分（G2），无先验知识的跟踪（G3）和评估质量得分（G4）来设计SiamFC++
    - G1 : 分类和状态估计任务的分离：分类任务将目标从干扰物和背景中分类出来，目标状态的估计如bbox回归等有利于对目标尺度变化的适应性
    - G2 : 得分无歧义：分类分数应该直接表示目标存在的置信度分数，即在“视野”中对应像素的子窗口，而不是预设置的锚点框。如SiamRPN容易产生假阳性结果，从而导致跟踪失败。
    - G3 : 无先验：不应该有目标比例/比率这样的先验知识，基于数据分发的先验知识阻碍了模型的泛化能力。
    - G4 : 预测质量的评估：直接使用分类置信度进行边框选择会导致性能下降，所以应该使用与分类无关的估计质量评分。目标状态评估分支（例如ATOM和DiMP：基于IoU-Net）的准确性很大程度上取决于此准则。

![](https://img-blog.csdnimg.cn/20200516221113774.png)

- 首个anchor-free跟踪器

## SiamMaskRCNN
- ArXiv2018
- 该文聚焦在一个前沿的问题：给一个包含了未知种类多个实体的没训练过的新样本(the query image)，如何检测以及分割所有这些实例
- 主要贡献在于
    - 1.提出siamese Mask R-CNN框架，能够仅给一个样本，就能够较好的检测&分割新的该样本同类实例
    - 2.构建了一个新的评测标准在MS-COCO

![](https://pic2.zhimg.com/80/v2-e226c65af54e79ac58a6e632df88a80d_1440w.jpg)

- 主要的4处不同已经用红色标识，即R、Siamese、Matching、L1
    - R代表了输入不仅有Query Image还有Reference Image
    - SiameseNetwork则对两者分别进行encode
    - Matching是将编码后的2个feature vector进行逐一的匹配
    - L1则是算diff的手段
- Matching流程如下图。融合特征和原始MRCNN编码的特征最大的不同在于包含了Ref和Scene双重信息

![](https://pic4.zhimg.com/80/v2-4c0555a703df4754b04027ba7e8d08a7_1440w.jpg)

## 总结思考
- 对于图像对匹配任务：SiamFC排除；SiamRPN、DaSiamRPN、SiamRPN++中，DaSiamRPN解决干扰物问题，在电商场景中不需考虑，另外两种由同一个框架实现，均可尝试；SiamFC++是anchor-free的，同意可以尝试，且开源代码比较新；SiamMask可以额外提供mask预测，属于锦上添花，暂时没有其他作用，暂保留

## Reference
- [1 github/pysot](https://blog.csdn.net/laizi_laizi/article/details/108279414)
- [2 SiamFC zhihu](https://zhuanlan.zhihu.com/p/107428605)
- [3 SiamFC代码解读CSDN](https://blog.csdn.net/weixin_39535701/article/details/113682500)
- [4 SiamRPN zhihu](https://zhuanlan.zhihu.com/p/110064574)
- [5 github/siammask](https://github.com/foolwood/SiamMask)
- [6 SiamMask zhihu](https://zhuanlan.zhihu.com/p/76460186?from_voters_page=true)
- [7 SiamMask author](https://zhuanlan.zhihu.com/p/58154634)
- [8 Siamese方法综述](https://zhuanlan.zhihu.com/p/66757733)
- [9 github/DaSiamRPN](https://github.com/foolwood/DaSiamRPN)
- [10 DaSiamRPN zhihu](https://zhuanlan.zhihu.com/p/42546692)
- [11 SiamFC++ CSDN](https://blog.csdn.net/qq_35078996/article/details/106165540)
- [12 github/SiamFC++](https://github.com/MegviiDetection/video_analyst)
- [13 SiamFC++ code analysis](https://blog.csdn.net/laizi_laizi/article/details/107157268)
- [14 SiamMaskRCNN zhihu](https://zhuanlan.zhihu.com/p/76820925)