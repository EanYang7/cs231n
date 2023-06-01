---
hide:
  - navigation
---
# [CS231n 卷积神经网络视觉识别](https://cs231n.github.io/)

[课程地址](http://cs231n.stanford.edu/)

这些笔记是与斯坦福大学计算机科学课程[CS231n：卷积神经网络视觉识别](http://cs231n.stanford.edu/)相关的。 如有问题/关注/漏洞报告，请直接向我们的[git repo](https://github.com/cs231n/cs231n.github.io)提交pull request。

---

<style type="text/css">
      .card {
        cursor: pointer;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
        transition: box-shadow 0.3s ease-in-out;
      }
      .card:hover {
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        color: green;
      }
      .card:active {
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
      }
      .card h2 {
        font-size: 20px;
        margin-bottom: 5px;
      }
      .card p {
        font-size: 16px;
        color: #666;
      }
      a {
        text-decoration: none;
      }
</style>

## 2023年春季作业

<a href="./assignments/2023/assignment1">

<div class="card">
  <h2>作业#1: 图像分类，kNN，SVM，Softmax，全连接神经网络Fully Connected Neural Network</h2>
</div>


</a>

<a href="./assignments/2023/assignment2">

<div class="card">
  <h2>作业#2: 全连接和卷积网络，批量归一化，dropout，Pytorch和网络可视化</h2>
</div>

</a><a href="https://www.example.com">

<div class="card">
  <h2>作业#3：使用RNN和Transformer进行图像字幕Image Captioning生成，网络可视化，生成对抗网络Generative Adversarial Networks，自监督对比学习 Self-Supervised Contrastive Learning</h2>
</div>


</a>

## 模块0：准备

<a href="setup">

<div class="card">
  <h2>软件设置</h2>
</div>


</a>

 <a href="python-numpy-tutorial">

<div class="card">
  <h2>Python / Numpy教程(使用Jupyter和Colab)</h2>
</div>


</a>

## 模块1：神经网络

<a href="classification">

<div class="card">
  <h2>图像分类：数据驱动方法，k-近邻k-Nearest Neighbor，训练/验证/测试分割</h2>
  <p>
   L1/L2距离，超参数搜索hyperparameter search，交叉验证cross-validation
  </p>
</div>


</a>

<a href="linear-classify">

<div class="card">
  <h2>线性分类：支持向量机SVM，Softmax</h2>
  <p>
参数化方法，偏差bias技巧，铰链损失hinge loss，交叉熵损失cross-entropy loss，L2正则化regularization，网络演示
  </p>
</div>


</a>

</a>

<a href="optimization-1">

<div class="card">
  <h2>优化：随机梯度下降Stochastic Gradient Descent</h2>
  <p>
优化景观landscapes，局部搜索，学习率，分析/数值梯度
  </p>
</div>


</a>

<a href="optimization-2">

<div class="card">
  <h2>反向传播Backpropagation，直觉Intuitions</h2>
  <p>
链式法则解释chain rule interpretation，实值电路real-valued circuits，梯度流模式patterns in gradient flow
  </p>
</div>


</a>

<a href="neural-networks-1">

<div class="card">
  <h2>神经网络第1部分：设置架构architecture</h2>
  <p>
生物神经元biological neuron模型，激活函数，神经网络架构，表示能力representational power
  </p>
</div>


</a>

<a href="neural-networks-2">

<div class="card">
  <h2>神经网络第2部分：设置数据和损失</h2>
  <p>
预处理，权重初始化，批量归一化batch normalization，正则化regularization (L2/dropout)，损失函数
  </p>
</div>


</a>

 <a href="neural-networks-3">

<div class="card">
  <h2>神经网络第3部分：学习和评估</h2>
  <p>
梯度检查，合理性sanity检查，看护babysitting学习过程，动量momentum（+ nesterov），二阶second-order方法，Adagrad / RMSprop，超参数hyperparameter优化，模型集成ensembles
  </p>
</div>


</a>

<a href="neural-networks-case-study">

<div class="card">
  <h2>将其组合起来：最小神经网络案例研究y</h2>
  <p>
最小2D玩具数据示例
  </p>
</div>


</a>

## 模块2：卷积神经网络

<a href="convolutional-networks">

<div class="card">
  <h2>卷积神经网络：架构，卷积/池化Pooling层</h2>
  <p>
层，空间排列spatial arrangement，层模式，层大小模式layer sizing patterns，AlexNet / ZFNet / VGGNet案例研究，计算考虑因素computational considerations
  </p>
</div>


</a>

<a href="understanding-cnn">

<div class="card">
  <h2>卷积神经网络的理解和可视化
</h2>
  <p>
tSNE嵌入，deconvnets，数据梯度，fooling ConvNets，人类比较human comparisons
  </p>
</div>


</a>

  <a href="transfer-learning">

<div class="card">
  <h2>迁移学习Transfer Learning和微调Fine-tuning卷积神经网络</h2>
</div>


</a>

## 学生投稿

  <a href="choose-project">

<div class="card">
  <h2>将课程项目转化为出版物</h2>
</div>


</a>

  <a href="rnn">

<div class="card">
  <h2>循环神经网络Recurrent Neural Networks</h2>
</div>

</a>

<html>
            <style>
            span:hover {
              transform: scale(1.5);
              color: lightblue;
            }
            }
          </style>

​              <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
​              <span class="badge badge--primary"><i class="fa fa-github" style="font-size: 150%;"></i> <a href="https://github.com/cs231n"> cs231n</span>
​            </html>
