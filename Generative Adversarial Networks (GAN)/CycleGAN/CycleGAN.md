# CycleGAN
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- CycleGAN可以完成GAN和CGAN的工作，可以从一个特定的场景模式图生成另外一个场景模式图，这两张场景模式中的物体完全相同。除此之外，CycleGAN还可以完成从一个模式到另外一个模式的转换，转换的过程中，物体发生了改变
- pix2pix的方法适用于成对数据的风格迁移，如下图左边。但是在大多数情况，对于A风格的图像，我们并没有与之相对应的B风格图像，我们所拥有的是一群处于风格A（源域）的图像和一群处于风格B（目标域）的图像，这样pix2pix2的方法就不管用了。CycleGAN的创新点在于能够在源域和目标域之间，无须建立训练数据间一对一的映射，就可实现这种迁移。这个方法的提出时间为2017年，目前来说是非常经典和基本的方法。

![](https://img-blog.csdnimg.cn/img_convert/09c436d43c6701ed395816dd1b45a37f.png)


- 如下图所示CycleGAN其实是由两个判别器(和)和两个生成器(G和F）组成，为了避免所有的X都被映射到同一个Y，所以为了避免这种情况，论文采用了两个生成器的方式，既能满足X->Y的映射，又能满足Y->X的映射，这一点其实就是变分自编码器VAE的思想，是为了适应不同输入图像产生不同输出图像。

![](https://img-blog.csdn.net/20180603093524366?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NDYyODQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


- （1）是判别器Y对X->Y的映射G的损失，判别器X对Y->X映射的损失也非常类似
- （2）是两个生成器的循环损失，这里其实是损失
- （3）是总损失
- （4）是对总损失进行优化，先优化D然后优化G和F，这一点和GAN类似
  
![](https://img-blog.csdn.net/20180603094002892?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NDYyODQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://img-blog.csdn.net/20180603103431263?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NDYyODQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://img-blog.csdn.net/20180603094349606?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NDYyODQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://img-blog.csdn.net/20180603094613600?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NDYyODQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


- [ref1](https://blog.csdn.net/qq_29462849/article/details/80554706)
- [ref2](https://blog.csdn.net/Mr_health/article/details/112545671)