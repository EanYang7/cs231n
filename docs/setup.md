---
layout: page
title: Software Setup
permalink: /setup-instructions/
---

今年，推荐的作业方式是通过 [Google Colaboratory](https://colab.research.google.com/)。但是，如果您已经拥有GPU支持的硬件，并且更喜欢在本地工作，我们将为您提供设置虚拟环境的说明。

- [在Google Colaboratory平台上远程工作](#working-remotely-on-google-colaboratory)
- [在您的机器上本地工作](#working-locally-on-your-machine)
  - [Anaconda虚拟环境](#anaconda-virtual-environment)
  - [Python venv](#python-venv)
  - [安装软件包](#installing-packages)

### 在Google Colaboratory平台上远程工作

Google Colaboratory总的来说是Jupyter notebook 和Google Drive的结合。它完全在云上运行，并预装了许多包（例如PyTorch和Tensorflow），因此每个人都可以访问相同的依赖项。更酷的是，Colab可以免费访问GPU（K80、P100）和TPU等硬件加速器，这对任务2和3特别有用。

**要求**。要使用Colab，您必须拥有一个带有相关Google Drive的谷歌帐户。假设两者都有，您可以通过以下步骤将Colab连接到您的Drive：

1. 单击右上角的控制盘，然后选择`设置`。
2. 单击`管理应用程序`选项卡。
3. 在顶部，选择`连接更多应用程序`，这将打开`GSuite Marketplace`窗口。
4. 搜索**Colab**，然后单击`添加`。

**工作流程**。每个作业都为您提供了一个下载链接，指向一个包含Colab笔记本和Python入门代码的zip文件。您可以将文件夹上传到云盘，在Colab中打开笔记本并对其进行操作，然后将进度保存回云盘。我们建议您观看下面的教程视频，其中以作业1为例介绍了推荐的工作流程。

<center><video src="./assets/video1.mp4" controls width="560" height="315" ></video></center>



 **最佳实践**。在与Colab合作时，您应该注意以下几点。首先要注意的是，资源并没有得到保证（这是免费的价格）。如果您空闲了一定时间，或者您的总连接时间超过了允许的最大时间（约12小时），Colab虚拟机将断开连接。这意味着任何未保存的进度都将丢失<font color="red"><strong>因此，养成在处理作业时经常保存代码的习惯</strong></font>。要了解更多关于Colab中资源限制的信息，请在[此处](https://research.google.com/colaboratory/faq.html)阅读他们的常见问题解答。

**使用GPU**。使用GPU就像在Colab中切换运行时一样简单。具体来说，`单击运行时->更改运行时类型->硬件加速器->GPU`，您的Colab实例将自动由GPU计算支持。

如果您有兴趣了解更多关于Colab的信息，我们鼓励您访问以下资源：

+ [谷歌Colab简介](https://www.youtube.com/watch?v=inN8seMm7UI)
+ [欢迎来到Colab](https://colab.research.google.com/notebooks/intro.ipynb)
+ [Colab功能概述](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

### 在您的机器上本地工作

如果您希望在本地工作，您应该使用虚拟环境。您可以通过Anaconda（推荐）或Python的本地`venv`模块安装一个。请确保您使用的是Python 3.7，因为**我们不再支持Python 2**。

#### Anaconda 虚拟环境

我们强烈建议使用免费的[Anaconda Python发行版](https://www.anaconda.com/download/)，为您提供了一种处理包依赖关系的简单方法。请务必下载Python 3版本，该版本目前安装的是Python 3.7。Anaconda的巧妙之处在于它默认情况下附带了[MKL优化](https://docs.anaconda.com/mkl-optimizations/)，这意味着您的`numpy`和`scipy`代码可以从显著的加速中获益，而无需更改一行代码。

一旦安装了Anaconda，就可以为课程创建一个虚拟环境。如果您选择不使用虚拟环境（强烈不推荐！），代码的所有依赖项都全局安装在您的机器上。要设置名为`cs231n`的虚拟环境，请在终端中运行以下操作：

```bash
# 这将在'path/to/anaconda3/envs/'中创建一个名为cs231n的anaconda环境
conda create -n cs231n python=3.7
```

要激活并进入环境，请运行`conda activate cs231n`。要停用环境，请运行`conda deactivate cs231n`或退出终端。请注意，每次您要处理任务时，都应该重新运行`conda activate cs231n`。

```bash
# 在激活环境之后，检查python二进制文件的路径是否与anaconda env的路径匹配
which python
# 例如，在我的机器上，打印出
# $ '/Users/kevin/anaconda3/envs/sci/bin/python'
```

您可以参考[本页](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)有关使用Anaconda管理虚拟环境的更详细说明。

**注意：**如果您选择走Anaconda 路线，您可以安全地跳过下一节，直接进入[安装软件包](#installing-packages)。

您可以参考[本页](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)有关使用Anaconda管理虚拟环境的更详细说明。

#### Python venv

从3.3开始，Python自带一个名为 [venv](https://docs.python.org/3/library/venv.html)的轻量级虚拟环境模块。每个虚拟环境都打包自己独立的一组已安装Python包，这些包与系统范围的Python包隔离，并运行与用于创建它的二进制文件相匹配的Python版本。要设置名为`cs231n`的虚拟环境，请在终端中运行以下操作：

```bash
# 这将在主目录中创建一个名为cs231n的虚拟环境
python3.7 -m venv ~/cs231n
```

要激活并进入环境，请运行`source~/cs231n/bin/activate`。要停用环境，请运行`deactivate`或退出终端。请注意，每次您想处理任务时，都应该重新运行`source~/cs231n/bin/activate`。

```bash
# 在激活环境之后，检查python二进制文件的路径是否与virtual env的路径匹配
which python
# 例如，在我的机器上，打印出
# $ '/Users/kevin/cs231n/bin/python'
```

#### 安装软件包

一旦您**设置了**并**激活了**您的虚拟环境（通过`conda`或`venv`），您就应该安装使用`pip`运行任务所需的库。要执行此操作，请运行：

```bash
# 同样，在运行以下命令之前，请确保您的虚拟环境（conda或venv）已被激活
cd assignment1  # 转到任务的目录

# 安装任务的依赖项
# 由于虚拟env被激活，该pip与环境的python二进制文件相关联
pip install -r requirements.txt
```
