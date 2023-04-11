# 如何安装PyTorch

## 安装Nvidia显卡驱动(可选)

对于带有nvidia显卡的机器，为了能让pytorch支持CUDA加速，我们需要安装显卡的驱动程序。

[驱动下载页面](https://www.nvidia.com/download/index.aspx)

驱动安装完成后，我们可以通过nvidia-smi命令来查看显卡状态。

![](https://content.lz1.fun/202304110840378.png)


## 创建虚拟环境

因为我们有时候需要运行不同版本的pytorch，所以我们不直接安装pytorch，而先创建虚拟环境，将pytorch安装到虚拟环境中。

```bash
conda create -n pytorch20 python=3.9
conda activate pytorch20
```

## 安装Pytorch

[PyTorch安装指引](https://pytorch.org/get-started/locally/)

Pytorch的安装非常简单，而且也集成了相应CUDA库的安装，我们只需要选择相应的计算平台，使用pip或conda命令进行安装即可。(不建议混用conda或者pip，即在创建好环境后，如果用pip安装pytorch，之后的均一直用pip；否则一直用conda.)


```bash
pip3 install torch torchvision torchaudio # Linux, pip, GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # Linux, pip, CPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # Linux, conda, GPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch # Linux, conda, CPU
```

安装好后，我们运行python，然后在python的交互式命令中输入

```python
import torch
x = torch.rand(5, 3)
print(x)
```

看是否能正常运行。