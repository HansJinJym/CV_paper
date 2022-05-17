## NMS

- 先对每个框的score进行排序，首先选择第一个，也就是score最高的框，它一定是我们要保留的框。然后拿它和剩下的框进行比较，如果IOU大于一定阈值，说明两者重合度高，应该去掉，这样筛选出的框就是和第一个框重合度低的框，第一次迭代结束。第二次从保留的框中选出score第一的框，重复上述过程直到没有框保留了

```python
def nms(dets, thresh):
    y1 = dets[:, 0]         # ymin
    x1 = dets[:, 1]         # xmin
    y2 = dets[:, 2]         # ymax
    x2 = dets[:, 3]         # xmax
    scores = dets[:, 4]     # confidence
    
    areas = (x2 - x1) * (y2 - y1)       # 每个boundingbox的面积
    order = scores.argsort()[::-1]      # boundingbox的置信度排序
    keep = []                           # 用来保存最后留下来的boundingbox
    while order.size > 0:     
        i = order[0]                    # 置信度最高的boundingbox的index
        keep.append(i)                  # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)
        
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep
```

## BN

![](https://img-blog.csdnimg.cn/20191217160803537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM5MzQ1Mg==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20191217160920586.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM5MzQ1Mg==,size_16,color_FFFFFF,t_70)

- 刚开始，$\beta$ 和 $\gamma$ 为0和1，然后再通过学习调整到合适的值
- 对于BN,在训练时，是对每一批的训练数据进行归一化，即用每一批数据的均值和方差，而在测试阶段，如进行一个样本的观测，并没有batch的概念，因此这个时候用的均值和方差是全部训练数据的均值和方差，可以通过移动平均法求得。

```python
def bn_forward_naive(x, gamma, beta, running_mean, running_var, mode = "trian", eps = 1e-5, momentum = 0.9):
	n, ic, ih, iw = x.shape
	out = np.zeros(x.shape)
	if mode == 'train':
		batch_mean = np.zeros(running_mean.shape)
		batch_var = np.zeros(running_var.shape)
		for i in range(ic):
			batch_mean[i] = np.mean(x[:, i, :, :]) 
			batch_var[i] = np.sum((x[:, i, :, :] - batch_mean[i]) ** 2 ) / (n * ih * iw)
		for i in range(ic):
			out[:, i, :, :] = (x[:, i, :, :] - batch_mean[i]) / np.sqrt(batch_var[i] + eps)
			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
		#update
		running_mean = running_mean * momentum + batch_mean * (1 - momentum)
		running_var = running_var * momentum + batch_var * (1 - momentum)
	elif mode == 'test':
		for i in range(ic):
			out[:, i, :, :] = (x[:, i, :, :] - running_mean[i]) / np.sqrt(running_var[i] + eps)
			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
	else:
		raise ValueError('Invalid forward BN mode: %s' % mode)
	return out
```