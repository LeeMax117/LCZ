# 1.任务解析
## 目标
此次任务本意为使用卫星图根据城市的热岛效应，来对未来的城市规划做一个指导预测。
>### 阶段一
第一阶段，按大赛要求，需要使用两颗卫星的卫星图，以及相应的分类，对其进行分析、训练，得到一个有效的模型，从而可以对未知的卫星图进行准确的预测。
>### 阶段二
**TODO**

# 第一阶段任务安排
## 1.第一周

   * 数据解析
      * **正确的读取数据**

		根据每个batch所需要的数据量，从hdf5格式数据中读取相应数据。

      * **对数据进行合理的处理**

		* 对sen1和sen2的数据进行合并，使用18个通道作为输入进行训练。
		* 根据整体数据集大小，生成一个打乱的perm列表，每次取数时，根据perm列表中存的index值，顺序取得数据，实现一个随机取数的功能。

      * 生成TFrecord
      	
		TODO


   * 建立模型
      * **选取合理的网络**

		目前采用的网络为两个卷积层两个池化层，一个全连接，使用validation集训练，batch_size 100的情况下，训练1000多个step可达到百分之70准确率。



      * **开始初步的训练**

		目前可以进行初步的训练，但网络结构，一些超参数，以及使用训练集进行训练等，都还急需解决。

## 2.第二周

   * save checkpoint
	   * 生成训练过程中的ckpt或pb，保存训练结果
   * summary
	   * 对关键监测点，如loss,lr，图像预处理后的结果等写相应的summary，以便在tensorboard中观察
   * inference
      * 完成inception程序
      * 根据测试集生成csv
      * 上传预测结果
   * 优化模型
      * 对比不同网络结构
      	* inception v4
      	* resnet
      	* densenet
      * 超参数优化
	  	* batch_size
	  	* leaning rate
	  	* bn
	  	* dropout
      * 选取合适的初始化权重及优化器
   * 数据预处理
      * **归一化**
      
		由于每个通道数据的取值范围不同，需要查阅相关资料，获得每个通道数据的取值范围后，将其取值化为（0，1）。

	  * **数据均衡**

		由于数据的分布未知，需要统计每个样本的数量，然后从每个样本中均匀取数，以实现样本均衡，从而达到更好的训练效果。

      * 筛选高质量数据
      * 数据增广
   * 数据读取优化
   * 如有必要的话，需租用GPU服务器
## 3.第三周

   * inception
      * 完成inception程序
      * 根据测试集生成csv
      * 上传预测结果
   * 继续对模型进行优化
   * 继续考虑更优化的预处理

***注：其中第二第三周任务可穿插进行，inception任务可随时根据生成的结点来刷新榜单***

# 2.任务中遇到的问题

## 通道问题

卫星图采用多频段的电磁波进行采样，与传统图像识别中的RGB三通道有一定差距

**讨论及研究结果**
>TODO

## 数据预处理问题
### 1.噪声影响

卫星图中不同频段的信息不同，有些高频波段图谱受云层影响较大，而有些其他频谱的也会受到另一些不明因素的影响，如何去除对训练结果有影响的数据。

**讨论及研究结果**
>TODO

>lyy：是否可以使用聚类或者是传统视觉算法的除燥方法，如卡尔曼滤波等对噪声进行处理？
### 2.裁剪翻转灰度调整等的影响

一般图像预处理过程中，会对图像进行裁剪拉伸，灰度调整等提高网络的泛化能力。在此次任务中，是否会出现某一分类进行某种变换之后就丧失了分类的特征，或变成另一种分类？

**讨论及研究结果**
>TODO

### 3.图像分辨率过低问题

**讨论及研究结果**
>TODO

>是否可以使用双线性插值或者学习一个反卷积等进行上采样来提高分辨率？

### 4.其他

***TODO***
## 模型选择

### 1.如何使用现有模型
首先，由于输入数据的shape与现有模型的差距很大，如何套用现有模型的结构而避免自己设计的模型可能出现各种问题。

其次，如何使用现有模型的权重进行fintune以节约训练时间提高模型准确率

**讨论及研究结果**
> TODO

### 2.网络结构的选择
可以选择ResNet、Inception V4、densenet等较新的分类网络。根据各个网络的特性来决定网络。

**讨论及研究结果**
> TODO

### 3.损失函数的选择
**讨论及研究结果**
> TODO

### 4.网络泛化能力及优化器等的选择
bn、dropout、learning_rate、batch_size、需要训练多少个epoch等

**讨论及研究结果**
> TODO

### 5.其他
**讨论及研究结果**
> TODO
