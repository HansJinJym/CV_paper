# Fast Online Object Tracking and Segmentation: A Unifying Approach

- 作者提出的模型，可以同时实现视频目标跟踪和视频目标分割这两个任务，并能达到实时的效果。作者称这个模型为SiamMask。该模型通过在用于目标跟踪的全卷积Siamese神经网络上增加mask分支来实现目标的分割，同时增强网络的loss，优化网络。一旦网络训练好之后，SiamMask仅依赖于初始的一个bounding box就可以实现类别无关的目标实时跟踪及分割（at 35 frames per second）。这个模型简单，功能多样，速度快，其效果也超越了其他跟踪方法。同时，还在DAVIS-2016, DAVIS-2017视频分割数据集上取得了具有竞争力的表现和最快的速度。
- SiamFC -> SiamRPN -> SiamMask

## Reference
- https://blog.csdn.net/weixin_43246440/article/details/99677258
- https://zhuanlan.zhihu.com/p/35040994