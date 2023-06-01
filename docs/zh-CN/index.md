---
hide:
  - navigation
---
# [CS231n 卷积神经网络视觉识别](https://cs231n.github.io/)

[课程地址](http://cs231n.stanford.edu/)

这些笔记是与斯坦福大学计算机科学课程\[CS231n：卷积神经网络视觉识别\](http://cs231n.stanford.edu/)相关的。 For questions/concerns/bug reports, please submit a pull request directly to our [git repo](https://github.com/cs231n/cs231n.github.io).

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

## Spring 2023 Assignments

<a href="./assignments/2023/assignment1">

<div class="card">
  <h2>Assignment #1: Image Classification, kNN, SVM, Softmax, Fully Connected Neural Network</h2>
</div>


</a>

<a href="./assignments/2023/assignment2">

<div class="card">
  <h2>Assignment #2: Fully Connected and Convolutional Nets, Batch Normalization, Dropout, Pytorch & Network Visualization </h2>
</div>

</a><a href="https://www.example.com">

<div class="card">
  <h2>Assignment #3: Network Visualization, Image Captioning with RNNs and Transformers, Generative Adversarial Networks, Self-Supervised Contrastive Learning</h2>
</div>


</a>

## Module 0: Preparation

<a href="setup">

<div class="card">
  <h2>Software Setup</h2>
</div>


</a>

 <a href="python-numpy-tutorial">

<div class="card">
  <h2>Python / Numpy Tutorial (with Jupyter and Colab)</h2>
</div>


</a>

## Module 1: Neural Networks

<a href="classification">

<div class="card">
  <h2>Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits</h2>
  <p>
   L1/L2 distances, hyperparameter search, cross-validation
  </p>
</div>


</a>

<a href="linear-classify">

<div class="card">
  <h2>Linear classification: Support Vector Machine, Softmax</h2>
  <p>
parameteric approach, bias trick, hinge loss, cross-entropy loss, L2 regularization, web demo
  </p>
</div>


</a>

</a>

<a href="optimization-1">

<div class="card">
  <h2>Optimization: Stochastic Gradient Descent</h2>
  <p>
optimization landscapes, local search, learning rate, analytic/numerical gradient
  </p>
</div>


</a>

<a href="optimization-2">

<div class="card">
  <h2>Backpropagation, Intuitions</h2>
  <p>
chain rule interpretation, real-valued circuits, patterns in gradient flow
  </p>
</div>


</a>

<a href="neural-networks-1">

<div class="card">
  <h2>Neural Networks Part 1: Setting up the Architecture</h2>
  <p>
model of a biological neuron, activation functions, neural net architecture, representational power
  </p>
</div>


</a>

<a href="neural-networks-2">

<div class="card">
  <h2>Neural Networks Part 2: Setting up the Data and the Loss</h2>
  <p>
preprocessing, weight initialization, batch normalization, regularization (L2/dropout), loss functions
  </p>
</div>


</a>

 <a href="neural-networks-3">

<div class="card">
  <h2>Neural Networks Part 3: Learning and Evaluation</h2>
  <p>
gradient checks, sanity checks, babysitting the learning process, momentum (+nesterov), second-order methods, Adagrad/RMSprop, hyperparameter optimization, model ensembles
  </p>
</div>


</a>

<a href="neural-networks-case-study">

<div class="card">
  <h2>Putting it together: Minimal Neural Network Case Study</h2>
  <p>
minimal 2D toy data example
  </p>
</div>


</a>

## Module 2: Convolutional Neural Networks

<a href="convolutional-networks">

<div class="card">
  <h2>Convolutional Neural Networks: Architectures, Convolution / Pooling Layers</h2>
  <p>
 layers, spatial arrangement, layer patterns, layer sizing patterns, AlexNet/ZFNet/VGGNet case studies, computational considerations
  </p>
</div>


</a>

<a href="understanding-cnn">

<div class="card">
  <h2>Understanding and Visualizing Convolutional Neural Networks
</h2>
  <p>
 tSNE embeddings, deconvnets, data gradients, fooling ConvNets, human comparisons
  </p>
</div>


</a>

  <a href="transfer-learning">

<div class="card">
  <h2>Transfer Learning and Fine-tuning Convolutional Neural Networks</h2>
</div>


</a>

## Student-Contributed Posts

  <a href="choose-project">

<div class="card">
  <h2>Taking a Course Project to Publication</h2>
</div>


</a>

  <a href="rnn">

<div class="card">
  <h2>Recurrent Neural Networks</h2>
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
