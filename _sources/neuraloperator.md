# Neural Operator

考虑
$$
\begin{equation}
\begin{array}{ll}
\mathcal{L} (u; \eta)(\bm x)=0, & x \in \Omega
\end{array}
\end{equation}
$$
这样的方程。我们称从参数(或者初边值)$\eta$到对应的解$u$的映射为解映射
$$
\begin{equation}
    \mathcal{S}: \eta \rightarrow u
\end{equation}
$$
Neural Operator 方法就是用神经网络去参数化这样一个解映射，然后使用optimize的方法从数据中去学习这个解映射。所以这类方法一般都是监督学习，即需要带标签的数据。

Neural Operator中使用的神经网络可以是任意网络，例如全连接网络、卷积神经网络，而Fourier Neural Operator{cite:p}`li2020fourier`是其中效果最好的网络之一。除了FNO，比较有名的此类方法还有DeepONet{cite:p}`lu2019deeponet`.

```{tableofcontents}
```