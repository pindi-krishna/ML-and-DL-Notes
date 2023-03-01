# Activation Functions

These are used to convert the linear input signals of a neuron into non-linear output signals to facilitate the learning of high order polynomials that go beyond one degree for deep networks. 

## Sigmoid
### Introduction

  $$ Sigmoid(x) = \frac{1}{1 + e^{-x}}$$
Sigmoid activation function range is $[0,1]$. Therefore, If the classification is between $0$ and $ 1$, Then use the output activation as sigmoid.

### Limitations

1. Gradient saturation
1. Computation expensive and Slow convergence 
1. Non-zero centered output that causes the gradient updates to propagate in varying directions


## Tanh :

$$ Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### Advantage
It produces a zero-centered output, thereby supporting the backpropagation process.

### Limitations

1. Gradient saturation
1. Computation expensive and Slow convergence.


## Softmax
### Introduction
This function is mainly used in multi-class models where it returns probabilities of each class, with the target class having the highest probability.
$$ Softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
i, j represents the different classes. 

## Rectified Linear Unit(ReLU)
$$ ReLU(x) = max(0,x) $$
     
### Advantages

1. Does not saturate
1. Computationally efficient
1. Converges mch faster than sigmoid/tanh in practive ( 6x)


### Limitations

1. Not zero centred output
1. If your unlucky, a neuron may be never active because the initialization has put it outside the manifold.
1. When the learning rate is high is easy to kill a lot of neurons. Imagine the activation function as a threshold which is moving while training. If the learning rate is to high it may move out of the data manifold. In that case the neuron dies and will never be able to recover because will never update again.

**It is a good practice to initialize them with a slightly positive initial bias to avoid ”dead neurons”**

## Leaky ReLU
$$                 
\begin{equation} 
\begin{split}
\text{Leaky-ReLU}(x) & = x \hspace{1.3cm} \text{if} x > 0  \\
 & =  \alpha.x \hspace{1cm} \text{if} x < 0
\end{split}
\end{equation}  $$
where $\alpha$ is a very small number (usually $0.01$) which can be a
if $(x > 0)$ hyperparameter or learned through. 

### Advantages

1. Does not saturate
1. Computationally efficient
1. Converges much faster than sigmoid/tanh in practice ( 6x)
1. Does not die


### Limitations

1. Not zero centred output
1. Consistency of the benefits across tasks not clear

## Q&A
## Why is it a problem if the outputs of an activation function are not zero-centered?

1. The output of every neuron in all the layers is always positive (between $0$ and $1$). During backpropagation, the derivative of loss w.r.t weights of contributing to the same neuron will always be either positive or negative. 

1. Thus, Movements of gradients is restricted to $1^{st}$ quadrant ($+$ and $+$) or $3^{rd}$ quadrant ($-$ and $-$).  As a result, the weight vector needs more updates to be trained properly, and the number of epochs needed for the network to get trained also increases. This is why the zero centered property is important, though it is NOT necessary.

1. Zero-centered activation functions ensure that the mean activation value is around zero. This property is important in deep learning because it has been empirically shown that models operating on normalized data––whether it be inputs or latent activations––enjoy faster convergence.

1. Unfortunately, zero-centered activation functions like $tanh$ saturate at their asymptotes –– the gradients within this region get vanishingly smaller over time, leading to a weak training signal.

1. ReLU avoids this problem but it is not zero-centered. Therefore all-positive or all-negative activation functions whether sigmoid or ReLU can be difficult for gradient-based optimization. So, to solve this problem deep learning practitioners have invented a myriad of Normalization layers (batch norm, layer norm, weight norm, etc.). we can normalize the data in advance to be zero-centered as in batch/layer normalization.
