****
A multi-stage data augment approach based on transfer learning algorithm. 

# Installation
The code is tested with Python 3.6, CUDA 10.1, Pytorch 1.5 on win10.

# Usage
## Dataset
* The links for the data we use are provided below:
    1. [WildFish](https://wildfish.in/)
    2. [F4K](http://groups.inf.ed.ac.uk/f4k/)

## Data Processing 
To train the model from scratch, use the following code:
```
 Part1 添加数据1.ipynb.py  # Alpha blending and Gaussian Fusion are carried out
 Part1 增加数据2.ipynb  # Add data to one folder
```

## Classifcation 
To train the model with transfer learning, use the following code:
```
 python 训练网络.py  # training the domain source
 Part2 使用ImageNet的预处理 对F4K 随机图片预测.ipynb  # The code for the random selected training
 Part2 使用mobileNetV2  训练F4K  挑选数据.ipynb  # The code for the data picked and image augmentation training
```

# Performance
 Training at source domain(/source domain performance.jpg)
 Training at target domain(/target domain performance.jpg)


