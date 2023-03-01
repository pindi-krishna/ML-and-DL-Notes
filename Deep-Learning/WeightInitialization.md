# Parameters Initialization for Deep Networks

1. [Medium Blog](https://medium.com/inveterate-learner/deep-learning-book-chapter-8-optimization-for-training-deep-models-part-ii-438fb4f6d135)
1. [deeplearning.ai Notes](https://rawgit.com/danielkunin/Deeplearning-Visualizations/master/initialization/index.html)

## WHY ?

1.  Initialization of parameters, if done correctly then optimization will be achieved in the least time otherwise converging to a minima using gradient descent will be impossible.

## Zero/Same Initialization

### Introduction

1. Initializing all parameters equally (zero or another value).
1.  Biases have no effect what so ever when initialized with $0$.

### Limitation

Initializing all the weights equally leads the neurons to learn the same weights during training. This causes symmetry breaking problem which means that if two hidden units are connected to the same input units, then these should have different initialization or else the gradient would update both the units in the same way and we don't learn anything new by using an additional unit. The idea of having each unit learn something different motivates random initialization of weights which is also computationally cheaper.

## Random Initialization

### Intuition

1. Biases are often chosen heuristically (zero mostly) and only the weights are randomly initialized, almost always from a Gaussian or uniform distribution. 

1. The scale of the distribution is of utmost concern. Large weights might have better symmetry-breaking effect but might lead to chaos (extreme sensitivity to small perturbations in the input) and exploding values during forward & back propagation. 

1. As an example of how large weights might lead to chaos, consider that there's a slight noise adding $\epsilon$ to the input. Now, we if did just a simple linear transformation like $W \times x$, the $\epsilon$ noise would add a factor of $W \times \epsilon$ to the output. In case the weights are high, this ends up making a significant contribution to the output.

1. If the weights are initialized to high values, the activations will explode and saturate to $1$. If the weights are initiliazed to small values, the activations will vanish and saturate to $0$. 


### Uniform Distribution

1.  Initialize weights from a uniform distribution with $n$ as the number of inputs in $l^{th}$ layer. 
    $$W_{i,j} \approx UniformDistribution(-\frac{1}{\sqrt{n^{l-1}}},\frac{1}{\sqrt{n^{l-1}}})$$


## Xavier Initialization

### Xavier Uniform

1.  Initialize weights from a uniform distribution. 
    $$W_{i,j} \approx UniformDistribution(-\frac{\sqrt{6}}{\sqrt{n^{l-1} + n^l}},\frac{\sqrt{6}}{\sqrt{n^{l-1} + n^l}})$$
1. Works well for sigmoid


### Xavier Normal

The goal of Xavier Initialization is to initialize the weights such that the variance of the activations are the same across every layer. This constant variance helps prevent the gradient from exploding or vanishing.
$$W_{ij} \approx  \mathcal{N}(0, \sigma^2)$$
$$\sigma^2 = \frac{2}{\sqrt{n^{l-1} + n^l}}$$
1. Assumptions : 

    1. Weights and inputs are centered at zero.
    1. Weights and inputs are independent and identically distributed.
    1. Biases are initialized as zeros.
    1. We use the tanh activation function, which is approximately linear with small inputs, (i,e) $Var(a^l) = Var(z^l)$

1. Xavier initialization is designed to work well with $tanh$ or $sigmoid$ activation functions.


## HE Initialization
### HE Uniform

1.  Initialize weights from a uniform distribution. 
    $$W_{i,j} \approx UniformDistribution(-\frac{\sqrt{6}}{\sqrt{n^{l-1}}},\frac{\sqrt{6}}{\sqrt{n^{l-1}}})$$
1. Works well for sigmoid


### HE Normal

1. Works well with ReLU.
$$W_{ij} \approx  \mathcal{N}(0, \sigma^2)$$
$$\sigma^2 = \frac{2}{\sqrt{n^{l-1}}}$$