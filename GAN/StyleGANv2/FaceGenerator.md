# AI Face

## Method: StyleGan
- Paper: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- Code: [Github](https://github.com/NVlabs/stylegan)
- Dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- Ref: [Zhihu](https://zhuanlan.zhihu.com/p/62119852)

## Open-source Projects
### Face Comparator
- 使用dlib库，提取人脸特征转换成numpy向量
- 对比向量之间的欧氏距离，即可计算出匹配度
- 33服务器 /home/sk49/new_workspace/jym/face_comparator
### Face Generator
- [this person does not exist](https://thispersondoesnotexist.com)
- Face generator based on [StyleGan](https://github.com/a312863063/seeprettyface-generator-yellow) and [StyleGan2](https://github.com/a312863063/generators-with-stylegan2)
- http://www.seeprettyface.com/
### Tricks
- [Changing face features](https://github.com/a312863063/Model-Swap-Face)，数字模特，同一个人脸进行风格替换，StyleGan+dlib

## 人脸贴图
- 流程：人脸定位 -> 识别关键点（6 points） -> 检测人脸方向
- 问题：人脸输入朝向、形状；人脸方向检测维度问题（参考b站《折磨先生》）

## AI美颜
- StyleGan 风格替换
- HSV空间转化 提亮美白、增加鲜艳度
- 高斯滤波器、双边滤波器 磨皮
- 卷积核 锐化
- 亮眼红唇

## Need to do
- 完善识别人脸朝向
- 具体美颜方法调研
- 新换脸思路尝试（dlib关键点扭曲）

## Reference
- https://zhuanlan.zhihu.com/p/29718304