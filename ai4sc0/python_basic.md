# Python 基础

Python本身的语法不复杂，直接看代码基本能看懂，所以这里不介绍。但是其实隐藏了很多坑，有空还是系统的看一下比较好，了解一下数据类型、控制语句、函数和类相关的东西。

[菜鸟教程](https://www.runoob.com/python/python-tutorial.html)

[官方文档](https://docs.python.org/zh-cn/3/)

## 在Windows/Linux上安装Python

这里不建议大家直接安装Python，而建议安装Anaconda（这是一个打包了一些常用科学计算库，以及虚拟环境管理器的Python发行版）

[Anaconda下载页面](https://www.anaconda.com/products/distribution)

对于Linux，下载安装脚本并bash安装。对于Windows，直接下载并双击安装。

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh
```

安装后打开终端。

## Python包管理

Python本身就是一门编程语言，支持的功能很简陋，我们之所以爱用Python是因为其丰富的第三方库。而Python的第三方库和C++需要自己编译不同，有统一的包管理软件：pip

我们常用的Python包有：

* [Numpy](https://numpy.org/)：Python的数值计算库，提供基本的向量、矩阵、张量操作。
* [Scipy](https://numpy.org/)：提供高阶算法，例如特殊函数、ODE求解等。
* [Matplotlib](https://matplotlib.org/)：仿matlab画图的库。
* [Pytorch](https://pytorch.org/)：可微科学计算库。（Numpy+Scipy+Differentiable）

Python自带的包管理软件就是pip，我们可以用pip方便的安装和卸载python包。

```bash
pip install numpy     # 使用 pip 安装 numpy
pip install numpy==1.0.0     # 使用 pip 安装特定版本 numpy
pip install numpy --user     # 使用 pip 安装 numpy 到个人目录 .pip 下
pip uninstall numpy          # 删除 numpy 包
```

## Python虚拟环境管理

