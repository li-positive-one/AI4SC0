# 大模型分布式训练



## 为什么我们需要机器学习的分布式训练？

![image-20230419091952898](https://content.lz1.fun/2023/04/19/c20a2f462765ae93a1e4182fc7793289-52750c.png)




- 模型规模迅速增加。2015年的 ResNet50 有2000万的参数， 2018年的 BERT-Large有3.45亿的参数，2018年的 GPT-2 有15亿的参数，而2020年的 GPT-3 有1750亿个参数。很明显，模型规模随着时间的推移呈指数级增长。目前最大的模型已经超过了1000多亿个参数。而与较小的模型相比，超大型模型通常能提供更优越的性能。图片来源: HuggingFace

- 内存效率： 训练万亿参数模型的内存要求远远超出了单个 GPU 设备中可用的内存要求。以混合精度使用 Adam 优化器进行训练需要大约 16 TB 的内存来存储模型状态（参数、梯度和优化器状态）。相比之下，最先进的NVIDIA A100 GPU只有40千兆字节（GB）的内存。它需要 400 个这样的 GPU 的集体内存来存储模型状态。
激活会消耗额外的内存，这些内存会随着批大小的增加而增加。仅使用单位批大小训练的万亿参数模型会产生超过 1 TB 的激活内存。激活检查点通过换取额外的计算将此内存减少到大约 20 GB，但内存要求对于训练来说仍然大得令人望而却步。
模型状态和激活必须在可用的多个 GPU 设备之间有效分区，以使此类模型甚至可以在不耗尽内存的情况下开始训练。

- 数据集规模迅速增加。对于大多数机器学习开发者来说，MNIST 和 CIFAR10 数据集往往是他们训练模型的前几个数据集。然而，与著名的 ImageNet 数据集相比，这些数据集非常小。谷歌甚至有自己的（未公布的）JFT-300M 数据集，它有大约3亿张图片，这比 ImageNet-1k 数据集大了近300倍。

- 计算能力越来越强。随着半导体行业的进步，显卡变得越来越强大。由于核的数量增多，GPU是深度学习最常见的算力资源。从2012年的 K10 GPU 到2020年的 A100 GPU，计算能力已经增加了几百倍。这使我们能够更快地执行计算密集型任务，而深度学习正是这样一项任务。

- 计算效率： 端到端训练一个万亿参数模型需要大约 5，000 个 zettaflops（即 5 个，后面有 24 个零;基于 OpenAI 的扩展工作定律）。训练这样一个模型需要 4，000 个 NVIDIA A100 GPUS，以 50% 的计算效率运行大约 100 天。
虽然大型超级计算 GPU 集群可以拥有超过 4，000 个 GPU，但由于批量大小限制，在这种规模下实现高计算效率具有挑战性。计算效率随着通信时间的增加而提高。此比率与批量大小成正比。但是，可以训练模型的批量大小有一个上限，超过上限，收敛效率会迅速下降。
世界上最大的模型之一 GPT-3 使用大约 1，500 的批量大小进行训练。对于 4，000 个 GPU，即使是 4，000 个的自由批处理大小也只允许每个 GPU 的批处理大小为 1，并限制了可扩展性。

如今，我们接触到的模型可能太大，以致于无法装入一个GPU，而数据集也可能大到足以在一个GPU上训练一百天。这时，只有用不同的并行化技术在多个GPU上训练我们的模型，我们才能完成并加快模型训练，以追求在合理的时间内获得想要的结果。



## 分布式计算

- Amdahl’s law
- Gustafson's law

### 通讯原语

![image-20230419092110089](https://content.lz1.fun/2023/04/19/174cbf3d9d17c1fa52184b3f86151273-d9167c.png)



## 分布式深度学习

<img src="https://content.lz1.fun/2023/04/19/b4747d8dee6b25963f42fcdf01f66aea-227351.png" alt="image-20230419092144203" style="zoom: 45%;" /><img src="https://content.lz1.fun/2023/04/19/5e87db76ccdbb474b8adfaad214b1561-30d474.png" alt="image-20230419092148651" style="zoom: 33%;" />



- 数据并行

![image-20230419092428125](https://content.lz1.fun/2023/04/19/a722723d56f85564d9ea4627fd99a481-4a927b.png)

- 张量并行

![image-20230419092443414](https://content.lz1.fun/2023/04/19/72c9e1e70ea4206629f6d18cc812d187-3aa821.png)

- 流水线并行

![image-20230419093231140](https://content.lz1.fun/2023/04/19/e470d925db8690e77333a077bf5fb16f-47ccb8.png)

- 混合并行

![image-20230419093239423](https://content.lz1.fun/2023/04/19/bb46c443b72332e43977603645c512f7-3859aa.png)

## 数据并行 

### torch.distributed

![image-20230419093407157](https://content.lz1.fun/2023/04/19/51868a12336aae6bcf4196a3bf12fff6-5cd5a4.png)



![image-20230419093419094](https://content.lz1.fun/2023/04/19/fffc8dc6ecff34fd412770a311d929b3-ae45c2.png)



### torch.nn.DataParallel

![image-20230419093441321](https://content.lz1.fun/2023/04/19/690608502770a1c5b25816aa83332620-09b050.png)



### Zero Redundancy Optimizer

- model states:
  - optimizer states (such as momentum and variances in Adam
  - Gradients
  - parameters. 
- remaining memory:
  - activation
  - temporary buffers 
  - unusable fragmented memory

[[1910.02054\] ](https://arxiv.org/abs/1910.02054)[ZeRO](https://arxiv.org/abs/1910.02054)[: Memory Optimizations Toward Training Trillion Parameter Models (arxiv.org)](https://arxiv.org/abs/1910.02054)

[[2101.06840\] ](https://arxiv.org/abs/2101.06840)[ZeRO](https://arxiv.org/abs/2101.06840)[-Offload: Democratizing Billion-Scale Model Training (arxiv.org)](https://arxiv.org/abs/2101.06840)

[[2104.07857\] ](https://arxiv.org/abs/2104.07857)[ZeRO](https://arxiv.org/abs/2104.07857)[-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (arxiv.org)](https://arxiv.org/abs/2104.07857)





#### ZeRO-DP Stage1 : Optimizer State Partitioning

#### ZeRO-DP Stage2 : Gradient Partitioning

#### ZeRO-DP Stage3 : Parameter Partitioning

#### ZeRO-R 

#### Communication Analysis of ZeRO-DP 

#### ZeRO-Offload



### PyTorch: FullyShardedDataParallel

![image-20230419093749565](https://content.lz1.fun/2023/04/19/1c87aded2b7ee0b1a750346c50f73a7d-20a20a.png)



## 张量并行

## Megatron-LM



## 流水线并行

### PyTorch: Pipeline (GPipe)





## 如何选择并行策略

[Efficient Training on Multiple GPUs (huggingface.co)](https://huggingface.co/docs/transformers/perf_train_gpu_many)

![image-20230419094208473](https://content.lz1.fun/2023/04/19/afb3ebb309bc701bf7bb96a7e81272bb-96dc60.png)



## 显存优化技巧

- Offload 
- Checkpointing   (T. Chen, 2016)
- 优化器 LARS, LAMB, Adafactor
- 梯度累加
- 混合精度