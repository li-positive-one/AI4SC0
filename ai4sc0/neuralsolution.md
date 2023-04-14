# Neural Solution

Neural solution是使用神经网络做为解的拟设，与之相对应的是使用分片常数、线性或者多项式作为解的拟设。然后，可以基于方程的强形式的残差，或者弱形式、变分形式等，得到优化问题，从而优化神经网络中的参数。

```{tableofcontents}
```

## DGM/PINN

DGM{cite:p}`sirignano2018dgm`和PINN{cite:p}`raissi2019physics`都是基于方程的残差实现的。考虑

$$
\begin{equation}
\begin{array}{ll}
\partial_{t} u(t, x)+\mathcal{L} {u}(t, x)=0, & ({t}, {x}) \in[0, {~T}] \times \Omega \\
{u}(0, { x})={u}_0({ x}), & { x} \in \Omega \\
{u}({t}, { x})={g}({t}, { x}), & { x} \in[0, {~T}] \times \partial \Omega
\end{array}
\end{equation}
$$

We first design a neural network as a fitting for the solution of this PDE, i.e., $u_{\theta}(t,x)$, where $\theta$ represents the trainable parameters of this neural network.

Then we randomly select a set of points within the interior of the computational region, as well as the region corresponding to the initial value condition and the region of the boundary boundary condition, respectively, i.e.

$$
\begin{equation}
\begin{aligned}
P_1&=\{x_i,t_i\}_{i=1}^{N_1} \in [0, {~T}] \times \Omega\\
P_2&=\{x_j,t_j\}_{j=1}^{N_2} \in [0, 0] \times \Omega\\
P_3&=\{x_k,t_k\}_{k=1}^{N_3} \in  [0, {~T}] \times \partial \Omega
\end{aligned}
\end{equation}
$$

Then we define the loss function

$$
\begin{equation}
\begin{aligned}
L(\theta,P_1,P_2,P_3)&=\sum_{(x,t)\in P_1}(u_{\theta}(t, x)-Lu(t,x))^2\\
&+\sum_{(x,t)\in P_2} (u_{\theta}(0, x)-u_0(x))^2\\
&+\sum_{(x,t)\in P_3} (u_{\theta}(t, x)-g(t,x))^2
\end{aligned}
\end{equation}
$$

It is easy to see that this loss function is actually an  approximation of

$$
\begin{equation}
\begin{aligned}
L(\theta)&=\int_{(x,t)\in P_1}(u_{\theta}(t, x)-Lu(t,x))^2 dx dt\\
&+\int_{(x,t)\in P_2} (u_{\theta}(0, x)-u_0(x))^2 dx dt\\
&+\int_{(x,t)\in P_3} (u_{\theta}(t, x)-g(t,x))^2 dx dt
\end{aligned}
\end{equation}
$$

obtained using Monte Carlo.

## Deep Ritz
In Deep ritz{cite:p}`yu2018deep`, the problem is a variational form of the equation, and we take the Poisson equation as an example to briefly introduce the process

$$
\begin{equation}
\begin{aligned}
-\Delta u(x)&=f(x)\quad &x &\in  \Omega\\
u(x)&=0 \quad &x &\in \partial \Omega  
\end{aligned}
\end{equation}
$$

Then we consider the variational form of this equation

$$
\begin{equation}
I(u)=\int_{\Omega}\left(\frac{1}{2}|\nabla u(x)|^2-f(x) u(x)\right) d x
\end{equation}
$$

Similarly, we use a neural network as an ansatz for the solution of this PDE and randomly select

$$
\begin{equation}
\begin{aligned}
P_1&=\{x_i,t_i\}_{i=1}^{N_1} \in \Omega\\
P_2&=\{x_j,t_j\}_{k=1}^{N_2} \in \partial \Omega
\end{aligned}
\end{equation}
$$

Then construct the loss function that

$$
\begin{equation}
I(u)=\int_{\Omega}\left(\frac{1}{2}\left|\nabla_x u(x)\right|^2-f(x) u(x)\right) d x+\beta \int_{\partial \Omega} u(x)^2 d s
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
L(\theta,P_1,P_2)&=\sum_{x \in P_1} \left(\frac{1}{2}\left|\nabla_x u(x)\right|^2-f(x) u(x)\right)  \\
&+\sum_{x\in P_2} (u_{\theta}( x))^2
\end{aligned}
\end{equation}
$$

This is followed by the same process of minimizing the loss function using the optimizer.

## 其他

除了使用残差或者变分形式解方程，将Neural Solution的思路和张量分解、幂法等传统方法结合，还可以做出很多有意思的工作，如{cite:p}`wang2022tensor`,{cite:p}`yang2022power`等。



```{bibliography}
:filter: docname in docnames
```