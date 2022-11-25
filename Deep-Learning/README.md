# Neural Networks

# Introduction

#### Neuron
Single neuron == linear regression without applying activation(perceptron). Followed by an activation function gives rise to linear/logistic regression. 

### Neural Network

**Simple Neural Network**

Each Input will be connected to the hidden layer and the NN will decide the connections. As the dataset size increases, NN performance increases.  

![NN](./Images/NN.jpg)

Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

![NNR](./Images/NNR.png)

In above figure ,Let $n_H$ = \# Hidden neurons, $n_X$ = \# input features, $n_y$ = \# output Classes.

1. Variables :
    
    1. $W_1 : (n_H, n_x) $, $b_1 : (n_H, 1) $
    1. $Z_1 : (n_H, 1) $, $a_1 : (n_H, 1) $
    1. $W_2 : (n_y, n_H)$, $b_2 : (n_y, 1) $
    1. $Z_2 : (n_y, 1) $,$a_2 : (n_y, 1) $
    
1. Forward Propagation : 
    
    1. $Z_1 = W_1 X + b_1 $
    1. $A_1 = g_1(Z_1)$
    1. $Z_2 = W_2 A_1 + b_2$
    1. $A_2 = Sigmoid(Z_2)$
1. Compute Cross Entropy (CE) Loss 
    $$L(A2, Y) = -Ylog(A2) - (1-Y)log(1-A2)$$    
1. Backward Propagation : Element wise product ($\times$)
    
    1. $dZ_2 = A2-Y$ 
    1. $dW_2 = \frac{1}{m}(dZ_2 \times {A_1}^T)$
    1. $db_2 = \frac{1}{m}\sum dZ_2$
    1. $dZ_1 = ({W_2}^T \times dZ_2) \times {g'}_1(Z_1) $  
    1. $dW_1 = \frac{1}{m} (d Z_1 \times X^T)$  
    1. $db_1 = \frac{1}{m}\sum dZ_1$
    

### Deep L-layer neural network

1. Shallow NN is a NN with one or two layers whereas Deep NN has three or more layers.

1. Variables :
    
    1. $n[l]$ is the number of neurons in a specific layer l. 
        1. $n[0]$ denotes the number of neurons input layer. 
        1. $n[L]$ denotes  the number of neurons in output layer.
    1. $g[l]$ is the activation function in $l$ layer.
    $a[l] = g[l](z[l])$.
    1. $w[l]$ weights is used for $z[l]$, shape : $(n[l], n[l-1])$
    1. $b[l]$ bias for layer $l$, shape : $(n[l],1)$
    1. $x = a[0]$ and  $y' = a[L]$ 
    
1. Forward Propagation : 
    $$Input  = A[l-1]$$
    $$ Z[l] = {W[l]}^T.A[l-1] + b[l]$$
    $$ A[l] = g[l](Z[l])$$
    $$ Output =  A[l]$$

1. Loss Function :  We generally use log-loss for classification.
   $$L = -y log(A) - (1-y)log(1-A) $$
   $$dA[L] = \frac{-y}{a}+\frac{1-y}{1-a} $$
   where $dA[L] = \frac{\partial L }{\partial A[L]}$
1. Backward Propagation : 
    $$Input =  da[l] $$
    $$dZ[l] = dA[l].g'[l](Z[l]) $$
    $$ dW[l] = \frac{1}{m} (dZ[l]{A[l-1]}^T)$$
    $$ db[l] = \frac{1}{m} sum(dZ[l])$$
    Dont forget $axis=1$ and $keepdims=True$ in the code.
    $$ Output = dA[l-1] = {w[l]}^T. dZ[l]$$ 
    
1. Algorithm :
    
    1. Initialize weights randomly
    1. Forward Propagation
    1. Compute Cost function.
    1. Compute gradients using Back Propagation
    1. Update all weights 
    1. Repeat above steps $ii-v$ till the weights converge (i.e.) till the lowest loss is achieved. 
    
1. Learnable Parameters : $W$ and $b$.
1. Hyper parameters (parameters that control the algorithm):
       
    1. Learning rate.
    1. Batch size
    1. Number of iteration.
    1. Number of hidden layers L.
    1. Number of hidden units n.
    1. Choice of activation functions.
    
    You have to experiment with different values of hyper parameters to find the best combination.

### Vanishing / Exploding gradients

1.  The Vanishing / Exploding gradients occurs when your derivatives become very small or very big.
1. Illustration : 
Suppose  that we have a deep neural network with number of layers $L$, and all the activation functions are linear and each $b = 0$. Then  
    $$ Y' = W[L]W[L-1].....W[2]W[1]X$$ 
1. Exploding Gradients :  If $W > I$ (Identity matrix) the activation and gradients will explode.
    $$W[l] = 
    \begin{bmatrix} 
    1.5 & 0 \\  
    0 & 1.5 \\ 
    \end{bmatrix}
    $$ 
    ($l != L$ because of different dimensions in the output layer)
    $$Y' = W[L]
    \begin{bmatrix}
    1.5 & 0 \\
    0 & 1.5 \\
    \end{bmatrix}^{L-1} X = 1.5^L$$ 
    which will be very large.
1. Vanishing Gradients : If W < I (Identity matrix) the activation and gradients will vanish.
    $$ W[l] = 
    \begin{bmatrix}
    0.5 & 0 \\
    0 & 0.5 \\
    \end{bmatrix}
    $$

    $$Y' = W[L]
    \begin{bmatrix}
        0.5 & 0 \\
        0 & 0.5 \\
        \end{bmatrix}^{L-1} 
        X = 0.5^L $$  
    which will be very small.
    
1. Recently Microsoft trained 152 layers (ResNet), which is a really giant network. With such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller, then gradient descent will take tiny steps. The model may learn something in a very long time or end up learning nothing if graidents tends to zero.

1. There is a partial solution that doesn't completely solve this problem but it helps a lot - careful choice of how you initialize the weights.

# Optimization
*This is notes which I made from Mithesh Khapra deep learning series.*
### Stochastic gradient descent (SGD)
#### Why
Batch gradient descent requires huge computation power when dealing with millions of samples because to make one update entire dataset gradients should be added.
#### How
Therefore, To reduce the computation power and time, we do the update for every datapoint. 
#### Problem
Too noisy regarding cost minimization because each point may updates/moves the parameters in a different direction. That is why, we need a method which is not noisy and requires less computation time and power. 

### Mini-batch gradient descent

1. Here, we do the update for every batch of data points. Usually batch size is less than $m$ and greater than $1$. 

1. Mini batch size is between $1$ and $m$. Training will be faster if batch size is power of $2$ (because of the way computer memory is layed out and accessed, sometimes your code runs faster if your mini-batch size is a power of $2$): $64, 128, 256, 512, 1024, ...$
1. Mini-batch gradient descent works much faster in case of large datasets.


#### Problem with Gradient descent
Wherever the error surface is steep, the loss decreases very quickly and wherever it is gentle, the loss decreases very slowly. To move on the gentle surface, usually it requires a lot of updates to be done. This increases the time complexity. Therefore, we need a method which moves very quickly on the gentle error surface also. 

### Gradient descent with momentum

#### Intuition

1. When moving on the error surface, we not only consider the gradient direction at that particular timestep but also consider the history i.e the gradient direction of previous timesteps. 

1. The gradient of the cost function at saddle points( plateau) is negligible or zero, which in turn leads to small or no weight updates. Hence, the network becomes stagnant, and learning stops.

1. If gradient direction at the previous timesteps and present timestep is same which is usually the case when moving on gentle surface, then we move by large amount. This is how, we can escape the gentle surface part quickly. 

1. Usually we give less importance to the gradient at the present time step and give more importance to history (gradients of previous timesteps). 
1. It also smoothens out the averages of skewed data points (oscillations w.r.t. gradient descent terminology). So this reduces oscillations in gradient descent and hence makes faster and smoother path towards minima.


#### Exponentially weighted averages

1. General equation :
    $$V_t = \beta \times V_{t-1} + (1-\beta) \times \nabla W_t$$
    given, $V_o = 0$, gradient at current time step $(\nabla W_t)$, History of previous gradients $V_{t-1}$. 
1. If we plot this, it will represent averages over $\approx \frac{1}{(1 - \beta)}$ entries:
    
    1. $\beta = 0.9$ will average last 10 entries
    1. $\beta = 0.98$ will average last 50 entries
    1. $\beta = 0.5$ will average last 2 entries
    
    Best $\beta$ average for our case is between 0.9 and 0.98 (Observed by AI researchers empirically).
1. Here we are giving more weight to the previous derivatives than the derivative of the current batch so that if the current batch has a lot of noise or turns out to contain more number of outliers, then all that noise will be given a less weightage. 

1. The momentum algorithm almost always works faster than standard gradient descent. The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights with the new values which will also reduce the noise. 


#### Update rule
$$V_t = \beta \times V_{t-1} + (1 - \beta) \times \nabla W_t$$
$$W_{t+1} = W_t - \eta \times V_{t+1}$$
given, $V_o = 0$ and $\eta$ - learning rate.  
#### Limitation


1. In GD with momentum, the update rule is  
    $$W_{t+1} = W_t - \underset{1^{st} move}{\eta \times \beta \times V_{t-1}} - \underset{2^{nd} move}{\eta \times (1-\beta)\nabla W_t}$$
1. While moving on the error surface, If the  gradient direction in the previous timesteps is similar as the present timesteps, then gradient picks up the momentum and it takes large steps and may overshoot. Then take u-turn, overshoot and again take u-turn, overshoot and this may go on for a while. 
1. Also, It may overshoot and enter another valley with local minima, where the gradient direction can completely be  changed and tries to converge at the local minima which shouldn’t be the case. 


### Nesterov accelerated gradient descent (NAG)
#### Intuition

1. In GD with momentum, the first move is due to the history and 2nd move is due to the gradient of $W$ of the current batch. So we know, that we are going to move by atleast by $\gamma.V_{t-1}$ and then a bit more by $\eta.\nabla W_t$. Then why not compute the gradient of $W_{look-ahead}$ at this paritally updated value of $W$ ($W_{look-ahead} = W_t - \gamma.V_t$) instead of computing $\nabla W_t$. Then do the update based on $\nabla W_{look-ahead}$.
1. Looking ahead helps NAG in correcting its course quicker than momentum based gradient descent. 


#### Update rule
$$W_{look-ahead} = W_t - \beta.V_{t-1}$$
$$V_t = \beta.V_{t-1} + (1- \beta).\nabla W_{look-ahead}$$
$$W_{t+1} = W_t - \eta \times V_t$$


### AdaGrad optimization algorithm
#### Intuition

1. In the real-world dataset, some features are sparse (for example, in Bag of Words most of the features are zero so it’s sparse) and some are dense (most of the features will be non-zero), so keeping the same value of learning rate for all the weights is not good for optimization. Some features gets updated many times and other features get updated fewer number of times. 

1. Hence, In case of sparse feature, this is imporatant and can't be ignored, then we would want to take the updates of that particular feature also very seriously. 

1. Therefore, can we have a different learning rate for each parameter which adapts with the frequency of feature updates. If the features is updated too many times, we decrease its learning rate by some amount and vice versa. This way we can allowe the movement of graidents in the direction of sparse features too.  

#### Update rule
Decay the learning rate for parameters in proportion to their update history. 
$$V_t = V_{t-1} + {(\nabla W_t)}^2$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t + \epsilon}} \times \nabla W_t$$
If any feature get too many updates, then learning rate of that feature is decreased and vice versa. 
#### Observation
In practice, this **does not** work so well if we remove the **sqaure root** from the denominator. (No idea why this happens).

#### Limitation

Over time the effective learning rate for $b$ will decay to an extent that there will be no further updates to $b$ (in the direction of $b$). 

Adagrad decays the learning rate very aggressively (as the denominator grows), as a result after a while the frequent parameters will start receiving very small updates because of the decayed learning rate.

### Root mean square prop(RMS Prop)
#### Intuition

1. The basic idea is that if there is one parameter in the neural network that makes the estimate of the cost function $J$ oscillate a lot, you want to penalize the update of this parameter during optimization, so to avoid the gradient descent algorithm adapt too quickly to changes in this parameter, as compared to the others.
1. To overcome the limitation of adagrad, why not decay the denominator and prevent its rapid growth. 
1. Hence, the normalization factor applied in RMSprop to update this parameter (i.e. its root mean square) will be larger compared to the rest of parameters, and the result of the normalization (and the update) smaller.

#### Update rule
Here, instead of considering all of the history, we consider some part of it ($\beta.V_{t-1}$). So, we are not agressively decreasing the learning rate based on the frequency of updates. 
$$V_t = \beta.V_{t-1} + (1-\beta){(\nabla W_t)}^2$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t + \epsilon}} \times \nabla W_t$$


### Adam optimization algorithm

#### Intuition
Adam optimization simply puts RMSprop and momentum together. 
#### Update rule

$$m_t = \beta_1.m_{t-1} + (1-\beta_1)\nabla W_t \text{\hspace{1cm} Momentum}$$  
$$V_t = \beta_2.V_{t-1} + (1-\beta_2){(\nabla W_t)}^2 \text{\hspace{1cm} RMS Prop}$$
$$	\hat{m_t} = m_t / (1 - {\beta_1}^t) \text{\hspace{2cm} Bias correction}$$   
$$	\hat{V_t} =  V_t / (1 - {\beta_2}^t) \text{\hspace{2cm} Bias correction}$$     
$$	W = W - \alpha \times \frac{\hat{m_t}}{\sqrt{\hat{V_t} + \epsilon}} $$

#### Intuition behind the bias correction
The reason we are doning this is that we dont't want to rely too much on the current gradient and instead rely on the overall behaviour of the gradients over many timesteps.
We are actually interested in the expected value of gradients and not on a single point estimate computed at time t. However, instead of computing $E[\nabla W_t]$ we are computing $m_t$ as the exponentially moving average. Hence, we would want $E[m_t] = E[\nabla W_t]$.
Therefore, we apply the bias correction because then the expected value of $\hat{m_t}$ is same as expected value of $\nabla W_t$.

#### Default Hyperparameters for Adam used in Practice

1. $\alpha$ : needed to be tuned.
1. $\beta_1$: parameter of the momentum - 0.9 is recommended by default.
1. $\beta_2$: parameter of the RMSprop - 0.999 is recommended by default.
1. $\epsilon: 10^{-8}$ is recommended by default.


### Learning rate decay

1. Slowly reduce learning rate.
1. As mentioned before mini-batch gradient descent won't reach the optimum point (converge). But by making the learning rate decay with iterations it will be much closer to it because the steps (and possible oscillations) near the optimum are smaller.
1. One technique equations is 
$$\alpha = (1 / (1 + decayrate \times epochnum)) \times \alpha_0$$
1. epoch-num is over all data (not a single mini-batch).
1. Other learning rate decay methods (continuous):
$$\alpha = (0.95 ^ {\text{epochnum}}) \times \alpha_0$$
$$\alpha = (k / \sqrt{epochnum}) \times \alpha_0 $$


# Activation Functions
These are used to convert the linear input signals of a neuron into non-linear output signals to facilitate the learning of high order polynomials that go beyond one degree for deep networks. 

### Sigmoid
#### Introduction

  $$ Sigmoid(x) = \frac{1}{1 + e^{-x}}$$
Sigmoid activation function range is $[0,1]$. Therefore, If the classification is between $0$ and $ 1$, Then use the output activation as sigmoid.

#### Limitations

1. Gradient saturation
1. Computation expensive and Slow convergence 
1. Non-zero centered output that causes the gradient updates to propagate in varying directions


### Tanh :

$$ Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

#### Advantage
It produces a zero-centered output, thereby supporting the backpropagation process.

#### Limitations

1. Gradient saturation
1. Computation expensive and Slow convergence.


### Softmax
#### Introduction
This function is mainly used in multi-class models where it returns probabilities of each class, with the target class having the highest probability.
$$ Softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
i, j represents the different classes. 

### Rectified Linear Unit(ReLU)
$$ ReLU(x) = max(0,x) $$
     
#### Advantages

1. Does not saturate
1. Computationally efficient
1. Converges mch faster than sigmoid/tanh in practive ( 6x)


#### Limitations

1. Not zero centred output
1. If your unlucky, a neuron may be never active because the initialization has put it outside the manifold.
1. When the learning rate is high is easy to kill a lot of neurons. Imagine the activation function as a threshold which is moving while training. If the learning rate is to high it may move out of the data manifold. In that case the neuron dies and will never be able to recover because will never update again.

**It is a good practice to initialize them with a slightly positive initial bias to avoid ”dead neurons”**

### Leaky ReLU
$$                 
\begin{equation} 
\begin{split}
\text{Leaky-ReLU}(x) & = x \hspace{1.3cm} \text{if} x > 0  \\
 & =  \alpha.x \hspace{1cm} \text{if} x < 0
\end{split}
\end{equation}  $$
where $\alpha$ is a very small number (usually $0.01$) which can be a
if $(x > 0)$ hyperparameter or learned through. 

#### Advantages

1. Does not saturate
1. Computationally efficient
1. Converges much faster than sigmoid/tanh in practice ( 6x)
1. Does not die


#### Limitations

1. Not zero centred output
1. Consistency of the benefits across tasks not clear

### Q&A
#### Why is it a problem if the outputs of an activation function are not zero-centered?

1. The output of every neuron in all the layers is always positive (between $0$ and $1$). During backpropagation, the derivative of loss w.r.t weights of contributing to the same neuron will always be either positive or negative. 

1. Thus, Movements of gradients is restricted to $1^{st}$ quadrant ($+$ and $+$) or $3^{rd}$ quadrant ($-$ and $-$).  As a result, the weight vector needs more updates to be trained properly, and the number of epochs needed for the network to get trained also increases. This is why the zero centered property is important, though it is NOT necessary.

1. Zero-centered activation functions ensure that the mean activation value is around zero. This property is important in deep learning because it has been empirically shown that models operating on normalized data––whether it be inputs or latent activations––enjoy faster convergence.

1. Unfortunately, zero-centered activation functions like $tanh$ saturate at their asymptotes –– the gradients within this region get vanishingly smaller over time, leading to a weak training signal.

1. ReLU avoids this problem but it is not zero-centered. Therefore all-positive or all-negative activation functions whether sigmoid or ReLU can be difficult for gradient-based optimization. So, to solve this problem deep learning practitioners have invented a myriad of Normalization layers (batch norm, layer norm, weight norm, etc.). we can normalize the data in advance to be zero-centered as in batch/layer normalization.

# Parameters Initialization for Deep Networkse

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


# Regularization
1. [Medium Blog](https://medium.com/inveterate-learner/deep-learning-book-chapter-7-regularization-for-deep-learning-937ff261875c)

1. [deeplearning.ai Notes](https://rawgit.com/danielkunin/Deeplearning-Visualizations/master/regularization/index.html)

## Introduction

1. The idea here is to limit the capacity of the model (the space of all possible model families) by adding a parameter norm penalty, $\Omega(\theta)$, to the objective function $J$. 
    $$\hat{J(\theta;X,y)}=J(\theta;X,y)+\lambda \Omega(\theta)$$
    Here, $\theta$ represents only the weights and **not the biases**. 
    1. $J(\theta;X,y)$ - Minimizes the training error of the model
    2. $\Omega(\theta)$ - Minimizes the complexity of the model. 
    
1. If your algorithm has a high variance :
    1. Try increasing data.
    1. Try regularization.
    1. Try using the less complex model. 

## $L1$  Regularization

$L1$ matrix norm is sum of absolute values of all weights of all layers $w_{ij}$.
$$||W|| = \sum ||w_{ij}||$$ 
The $L1$ regularization Loss:
$$J(w,b) =  \frac{1}{m} \times \left[\sum L(y(i),y'(i)) \right] + \lambda \times ||W||$$

$L1$ regularizer can drive the weights of irrelevant features to zero. Therefore, it is also used for feature selection. 

## $L2$ Regularization

$L2$ matrix norm also called as Frobenius norm is sum of squares of all weights of all layers $w_{ij}$ .
$$||W||^2 = \sum||w_{i,j}||^2 = W^T.W $$
The L2 regularization Loss:
$$J(w,b) =  \frac{1}{m} \times \left[ \sum (L(y(i),y'(i))) \right] + \lambda \times ||W||^2$$
In most cases, $L2$ regularization is being used. $L2$ penalizes the outliers more than $L1$. In practice this penalizes large weights and effectively limits the freedom in your model (complexity of the model).

### Why don't we regularize the bias ?

1. We usually do not regularize the bias, since it generally has a lower variance than the weights, due to the bias not interacting with both the inputs and the outputs, as the weights do.

1. For example for a linear model $f(x)=mx+b$, the bias term $b$ allows us to fit lines that do not pass through the origin - However if we regularize bias term, we are encouraging the bias to stay close to $0$, which defeats the purpose of the bias term in the first place. You can read more about this [here](https://www.deepwizai.com/simply-deep/why-does-regularizing-the-bias-lead-to-underfitting-in-neural-networks).

### Why regularization reduces overfitting ? 

Here are some intuitions:

1. $L2$ regularization retains the weights corresponding to features with high eigen values and scales down significantly the weights of the features with low eigen values. (This is what usually happens in PCA. Features with low eigen values in the covariance vector are removed). [Proof](https://www.youtube.com/watch?v=lg4OLAjxRcQ&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU&index=62) 

1. In general, whenever we train a neural network, we give a lot of freedom to weights. So, weights can take any real value and try to make the training error zero (overfitting). Therefore, we try to restrict the weights to between certain range, or penalize its magnitude so that it doesn't overfit the training data and generalize well on the test data. 

1. If $\lambda$ is too large - a lot of $w$'s will be close to zeros which will make the $NN$ simpler (you can think of it as it would behave closer to logistic regression).
If $\lambda$ is good enough it will just reduce the weights that makes the neural network overfit.

## Data Augmentation

1. Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.

1. Data augmentation techniques such as cropping, padding, and horizontal $\&$ vertical flipping are commonly used to train large neural networks.

1. For a classification task, we desire for the model to be invariant to certain types of transformations, and we can generate the corresponding $(x,y)$ pairs by translating the input $x$.

1. However, caution needs to be mentioned while data augmentation to make sure that the class doesn't change. For e.g., if the labels contain both $"b"$ and $"d"$, then horizontal flipping would be a bad idea for data augmentation. Add random noise to the inputs is another form of data augmentation, while adding noise to hidden units can be seen as doing data augmentation at multiple levels of abstraction.

1. Finally, when comparing machine learning models, we need to evaluate them using the same hand-designed data augmentation schemes or else it might happen that algorithm $A$ outperforms algorithm $B$, just because it was trained on a dataset which had more / better data augmentation.


## Noise Robustness
### Adding Noise to the input

1. Usually, we corrupt the original image/input by adding gaussian noise (sampled from gaussian distribution). Similar to data augmentation. 

1. Adding gaussian noise to the input works similar to $L2$ regularization. Dimension specific scaling of weights will happen, which scales down the non-important weights significantly.  [Mathematical proof can be found](https://www.youtube.com/watch?v=agGUR06jM_g&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU&index=65).

    $$\hat{x_i} = x_i + \epsilon_i$$
    $$\hat{y} = \sum w_i x_i$$
    $$\tilde{y} = \sum w_i \hat{x_i}$$

    $\epsilon$ -- small values drawn from gaussian distribution.

1. Usually, our goal is to minimze the squared error $E(\tilde{y} - y)^2$. If we substitute $\tilde{y}$ in the squared error equation and try to get $E(\hat{y} - y)^2$ outside, then equation will end up in the form of regularization.
    $$\hat{J(\theta;X,y)}=J(\theta;X,y)+\lambda \Omega(\theta)$$


### Adding noise to weights

Noise can even be added to the weights. Now, suppose a zero mean unit variance Gaussian random noise, $\epsilon$, is added to the weights. We still want to learn the appropriate mapping through reducing the mean square. Minimizing the loss after adding noise to the weights is equivalent to adding another regularization term which makes sure that small perturbations in the weight values don’t affect the predictions much, thus stabilising training.

(Need to elaborate more)

### Label Smoothing

Label Smoothing is a regularization technique that introduces noise for the labels. **This accounts for the fact that datasets may have mistakes in them**, so maximizing the likelihood of $log p(y|x)$ directly can be harmful. In such a case, we can add noise to the labels by assigning a probability of $(1-\epsilon)$ that the label is correct and a probability of $\epsilon$ that it is not. In the latter case, all the other labels are equally likely. Label Smoothing regularizes a model with $k$ softmax outputs by assigning the classification targets with probability $(1-\epsilon)$ or choosing any of the remaining $(k-1)$ classes with probability $\epsilon/ (k-1)$.

(Need to elaborate more)
## Early stopping

1. In this technique we plot the training set and the validation set cost together for each iteration. At some iteration the validation set cost will stop decreasing and will start increasing.

1. We will pick the point at which the training set error and validation set error are best (lowest training cost with lowest validation cost). We will take these parameters as the best parameters.

1. However, since we are setting aside some part of the training data for validation, we are not using the complete training set. So, once Early Stopping is done, a second phase of training can be done where the complete training set is used. There are two choices here:

1. Train from scratch for the same number of steps as in the Early Stopping case.
1. Use the weights learned from the first phase of training and retrain using the complete data.

1. Other than lowering the number of training steps, it reduces the computational cost also by regularizing the model without having to add additional penalty terms. 

1. It allows only $t$ updates to the parameters. It affects the optimization procedure by restricting it to a smal volume of the parameter space, in the neighbourhood of the initial parameters ($\theta$).

1. Early stopping is another way of regularization. But, it also can be used along with regularization ($L2$ or $L1$).

1. Under certain assumptions, this is similar to L2 regularization. [Mathematical Proof can be found here](https://www.youtube.com/watch?v=zm5cqvfKO-o&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU&index=67).


## Parameter Tying and Parameter Sharing

1. Till now, most of the methods focused on bringing the weights to a fixed point. However, there might be situations where we might have some prior knowledge on the kind of dependencies that the model should encode. 
1. Suppose, two models $A$ and $B$, perform a classification task on similar input and output distributions. In such a case, we'd expect the parameters ($W_a$ and $W_b$) to be similar to each other as well. We could impose a norm penalty on the distance between the weights, but a more popular method is to force the set of parameters to be equal. This is the essence behind Parameter Sharing. 
1. A major benefit here is that we need to store only a subset of the parameters (e.g. storing only $W_a$ instead of both $W_a$ and $W_b$) which leads to large memory savings.

## Bagging

### Introduction
The techniques which train multiple models and take the maximum vote across those models for the final prediction are called ensemble methods. The idea is that it's highly unlikely that multiple models would make the same error in the test set.

### Algorithm

1. Train multiple independent models.
1. At test time average their results or take the majority vote.

### Limitation
Computationally very expensive for both training and testing.

## Dropout Regularization

### Why ?
The problem with bagging is that we can't train an exponentially large number of models and store them for prediction later. Dropout makes bagging practical by making an inexpensive approximation. In a simplistic view, dropout trains the ensemble of all sub-networks formed by randomly removing a few non-output units by multiplying their outputs by $0$.

### Introduction

1. Dropout refers to dropping out units/neurons. Temporarily remove a node and all its incoming/outgoing connections resulting in a thinned network. 
1. Each node/neuron is retained with a fixed probability (usually $0.5$) for hidden nodes and $p = 0.8$ for visible nodes. 
1. Tricks: Share the wrights across all the networks and samples a different network for each training instance. 
1. If there are $n$ neurons, then $2^n$ networks are possible to train. Each thinned network gets trained rarely (or even never) but the **parameter sharing ensures that no model has untrained or poorly trained parameters**.

1. **At test time**, it is impossible to aggregate the output of $2^n$ thinned networks. Instead we use the full neural network and **scale the output of each node by the fraction of times it was on during training.** 

1. Dropout prevents hidden neurons form co-adapting. Essentially a hidden neuron cannot rely too much on other neurons as they may get dropped out any time. Each hidden neuron has to learn, to be more robust to these random dropouts. 
1. Reduces the representational capacity of the model and hence, the model should be large enough to begin with. 
1. Works better with more data.

### Algorithm
In case of dropout, The forward propagation becomes 
$$Input  = A[l-1]$$
$$ Z[l] = {W[l]}^T.A[l-1] + b[l]$$
$$ A[l] = g[l](Z[l])$$
$$ Output =  D[l] (A[l])$$

Where $D$ is the dropout layer (Masking the activations $D[l] \in {0,1}$). The key factor in the dropout layer is $keep\_prob$ parameter, which specifies the probability of keeping each unit. Say if $keep\_prob = 0.8$, we would have $80\%$ chance of keeping each output unit as it is, and $20\%$ chance set them to $0$. 

Code snippet would be like: 

    keep_prob = keep_probs[i-1]
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)  # Generating the Mask D
    A = np.multiply(D, A)  # Masking

### Differences between bagging and Dropout

1. In bagging, the models are independent of each other, whereas in dropout, the different models share parameters, with each model taking as input, a sample of the total parameters.
1. In bagging, each model is trained till convergence, but in dropout, each model is trained for just one step and the parameter sharing makes sure that subsequent updates ensure better predictions in the future


### Intuition
Adding noise in the hidden layer is more effective than adding noise in the input layer. For e.g. if some unit $h_i$ learns to detect a nose in a face recognition task. Now, if this $h_i$ is removed, then some other unit either learns to redundantly detect a nose or associates some other feature (like mouth) for recognising a face. In either way, the model learns to make more use of the information in the input. On the other hand, adding noise to the input won't completely removed the nose information, unless the noise is so large as to remove most of the information from the input.

### Dropout at Test time
We don't turn off any neurons during test time. We consider all the neurons but only with  probability $p$, as each and every neuron is active only with probability $p$, during training time. We multiply its activation value only with probablity $p$ to get the predictions.

(OR)

While training, the first term can be approximated in one pass of the complete model **by dividing the weight values by the keep probability (weight scaling inference rule)**. The motivation behind this is to capture the right expected values from the output of each unit, i.e. the total expected input to a unit at train time is equal to the total expected input at test time. A big advantage of dropout then is that it doesn’t place any restriction on the type of model or training procedure to use. Therefore, if we divide the weight values  by $p$, then we need not do anything while inference. We can directly use the full NN as it is. 

# Auto Encoders
[Source](https://gabgoh.github.io/ThoughtVectors/)
## Introduction

### Latent Vector or Thought Vector
Neural networks have the rather uncanny knack for turning meaning into numbers. Data flows from the input to the output, getting pushed through a series of transformations which process the data into increasingly abstruse vectors of representations. These numbers, the activations of the network, carry useful information from one layer of the network to the next, and are believed to represent the data at different layers of abstraction.
Thought vectors have been observed empirically to possess some curious properties. 

![latent_smile](./Images/Latent_Smile.png)

### The encoder of a linear autoencoder is equivalent to PCA if:

1. Use a linear encoder.
1. Use a linear decoder.
1. Use mean squared error loss function. 
1. Standardize the inputs. 


## Regularization in auto-encoders

### Intuition

$$Total Loss = L(\theta) + \Omega(\theta)$$

1. $L(\theta)$ -- Prediction error. Minimizing this will capture important variations in data. 
2. $\Omega(\theta)$ -- Regularization error. Minimizing this do not capture variations in data. 
3. Tradeoff -- Captures only very important variations in data. 

### Tying weights
This is one of the trick to reduce over-fitting. We enforce $W^* = W^T$. Here, we have only one $W$ to update. 
During the back propagation, w.r.t decoder, we find $\partial L / \partial W$ as we usually do. 

**However, we add derivative computed w.r.t decoder to derivative w.r.t encoder to find the final derivative w.r.t encoder.**  

### Denoising auto-encoders

We corrupt the data with some low probability which usually we encounter in the test data and pass that corrupted input to the auto-encoder and try to reconstruct the original data. 

#### Intuition

1. The model gets robust to the expected noise in the test set and perform well on test set too. 
1. For ex, in case of BMI prediction using height, weight parameters, if we corrupt height, then to reconstruct the original height (without corruption), the AE model should learn the interactions between height and weights so that it can figure out what went wrong and correct the corruption. 

Empirically, it has been found that, In case of gausssian noise added AE filters captures more meaningful patterns like edges, corners than simple L2 regularized AE. 

### Sparse Auto-encoders

We use a different type of regularization technique to restrict the freedom of weights. In sparse AE, we ensure that the neuron is inactive most of the times. 

The constraint imposed is on the average value of the activation of neurons. The average activation of neurons in layer $l$ is given by 
$$\hat{\rho} = \frac{1}{k}\sum_i^k {h(x_i)}_l$$
where $k$ is the number of the neurons in the layer $l$. 
We are trying to keep the average value of the activations of neurons to be close to small value ($\rho = 0.005$).

#### Intuition: 

We are trying to prevent the neuron from firing most of the times by enforcing the above constraint . As the neuron fires very few number of times, it tries to learn meaningful patterns during training to reduce the training error. 

#### How ? 
$$\Omega(\theta) = \sum_{i=1}^k \rho log\frac{\rho}{\hat{\rho}} + (1-\rho) log\frac{1- \rho}{1 - \hat{\rho}}$$

### Contractive auto-encoders
[Source](https://iq.opengenus.org/contractive-autoencoder/)
#### Introduction
1. Contractive autoencoder simply targets to learn invariant representations to unimportant transformations for the given data.

1. It only learns those transformations that are provided in the given dataset so it makes the encoding process less sensitive to small variations in its training dataset.

1. The goal of Contractive Autoencoder is to reduce the representation’s sensitivity towards the training input data.

### Formula
1. Frobenius norm of the Jacobian matrix of hidden activations. 
    $$\Omega(\theta) = ||{J_x(h)||_F}^2$$
    $$J_x(h) = \sum_{j=1}^n \sum_{l=1}^k {\left(\frac{\partial h_l}{\partial x_j} \right)}^2$$
    $h_l$ represents the latent representation learnt by the last layer of encoder. 
1. If this value is zero, it means that as we change input values, we don't observe any change on the learned hidden representations.

1. But if the value is very large, then the learned representation is unstable as the input values change.

# Practical Aspects of Deep Learning

### Train/Validation/Test sets

1.  Its impossible to get all your hyperparameters right on a new application from the first time. So the idea is you go through the loop: Idea ==> Code ==> Experiment. You have to go through the loop many times to figure out your hyperparameters.
1. Your data will be split into three parts:
    * Training set. (Has to be the largest set)
    * Hold-out cross validation set / Development or "dev" set.
    * Testing set.
1. You will try to build a model upon training set then try to optimize hyperparameters on validation set as much as possible. Then after your model is ready, you try and evaluate the testing set.
1.  So, the general trend on the ratio of splitting the models:
    
    1. If size of the dataset is $100$ to $1000000$ ==> $60/20/20$
    1.  If size of the dataset is $1000000$ to $\inf$ ==> $98/1/1$ or $99.5/0.25/0.25$.

1. Make sure the validation and test set are coming from the same distribution.
For example, if cat training pictures is from the web and the validation/test pictures are from users cell phone, then they will mismatch. It is better to make sure that validation and test set are from the same distribution.
1.  The training error will generally increase as you increase the dataset size. This is because the model will find it harder to fit to all the datapoints exactly. Also, by increasing the dataset size, your validation (dev) error will decrease as the model would learn to be more generalized.


### Normalizing inputs

1. Normalizing a set of data transforms the set of data to be on a similar scale. For machine learning models, our goal is usually to recenter and rescale our data such that is between 0 and 1 or -1 and 1, depending on the data itself. 
    $$f(x) = \frac{x - \hat{x}}{\sigma}$$
    where $x$ represents a feature or attribute , $\hat{x}$ represents the mean of $x$ and $\sigma$ represents the standard deviation of $x$
1. Normalization can help training of our neural networks as the different features are on a similar scale, which helps to stabilize the gradient descent step, allowing us to use larger learning rates or help models converge faster for a given learning rate.

### Shuffling the dataset

#### Should we shuffle the dataset before training ?

The obvious case where you'd shuffle your data is if your data is sorted by their class/target. Here, you will want to shuffle to make sure that your training/test/validation sets are representative of the overall distribution of the data. The idea behind batch gradient descent is that by calculating the gradient on a single batch, you will usually get a fairly good estimate of the "true" gradient. That way, you save computation time by not having to calculate the "true" gradient over the entire dataset every time.

#### Should you shuffle the dataset after every epoch ?

[Source](https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks)
1. Shuffling data serves the purpose of reducing variance and making sure that models remain general and overfit less.
1. You want to shuffle your data after each epoch because you will always have the risk to create batches that are not representative of the overall dataset, and therefore, your estimate of the gradient will be off. Shuffling your data after each epoch ensures that you will not be "stuck" with too many bad batches.

1. In regular stochastic gradient descent, when each batch has size $1$, you still want to shuffle your data after each epoch to keep your learning general. Indeed, if data point $17$ is always used after data point $16$, its own gradient will be biased with whatever updates data point $16$ is making on the model. 

1. By shuffling your data, you ensure that each data point creates an "independent" change on the model, without being biased by the same points before them.

## Batch Normalization

[Source](https://mmuratarat.github.io/2019-03-09/batch-normalization)

### Why

Typically, we train model in mini-batches. Initially, we make sure the input layer is zero centered and unit gaussian. But in the deep network, there is a possibility that the distributions of hidden layers may change which makes the learning process hard. That's why, Why not ensure that the pre-activations at each layer have similar distributions which makes the learninig easy. 

### Introduction

Batch Normalization will let the network decide what is the best distributions for it instead of forcing it to have zero mean and unit variance. This is usually applied with mini-batches. This is a co-variate shift. Training the model on different possible distributions of data. 

### Parameters
1. $\mu_\phi \in R^{1 \times D}$ : is the empirical mean of each input dimension across the whole mini-batch.
1. $\sigma_\phi \in R^{1 \times D}$ 
 is the empirical standard deviation of each input dimension across the whole mini-batch. N is the number of instances in the mini-batch
1. $\hat{x_i}$ is the zero-centered and normalized input.
1. $\gamma \in R^{1 \times D}$ is the scaling parameter for the layer.
1. $\beta \in R^{1 \times D}$
 is the shifting parameter (offset) for the layer.
1. $\epsilon$ is added for numerical stability, just in case $\sigma_\phi^2$ turns out to be $0$ for some estimates. This is also called a smoothing term.
1. $y_i$ is the output of the BN operation. It is the scaled and shifted version of the inputs. 
$y_i=BN_{\gamma, \beta}(xi)$.

### Algorithm
For a given input batch $\phi$ of size $(N,F)$ going through a hidden layer of size $D$, some weights $w$ of size $(F,D)$ and a bias $b$ of size $(D)$, we first do an affine transformation $X=Z⋅W+b$ where $X$ contains the results of the linear transformation (size $(N,D)$).
Note that, all the expressions above implicitly assume broadcasting as $X$ is of size $(N,D)$ and both $\mu_\phi$ and ${\sigma^2}_\phi$ have size equal to $D$.

$$\hat{x_{il}}=\frac{x_{il} - \mu_{\phi l}}{\sqrt{{\sigma^2}_{\phi l} + \epsilon}}$$
where
$$\mu_{\phi l}=\frac{1}{N} \sum_{p=1}^N x_{pl}$$
and
$${\sigma^2}_{\phi l}=\sum_{p=1}^N \frac{1}{N} (x_{pl} - \mu_{\phi l})^2$$
with $i=1…,N$ and $l=1,…,D$.

Every component of $\hat{x_i}$ has zero mean and unit variance. However, we want hidden units to have different distributions. In fact, this would perform poorly for some activation functions such as the sigmoid function. Thus, we’ll allow our normalization scheme to learn the optimal distribution by scaling our normalized values by $\gamma$ and shifting by $\beta$.

$$y_i \leftarrow \gamma⋅\hat{x_i} + \beta ≡ BN_{\gamma,\beta}(x_i) \quad \text{(scale and shift)}$$
**Note**
1. $\gamma$ and $\beta$ are learnable parameters that are initialized with $\gamma=1$ and $\beta=0$.
1. Batch Normalization is done individually at every hidden unit. **The pairs ($\gamma^k$, $\beta^k$) are learnt per neuron**

### Batch normalization at test time

In testing we might need to process examples one at a time. The mean and the variance computed for training batches won't make sense to use in case of of one test sample. We have to compute an estimated value of mean and variance to use it in testing time. 

Exponentially weighted “moving average” can be used as global statistics to update population mean and variance:
$$\mu_{mov} = \alpha \mu_{mov}+(1-\alpha)\mu_\phi$$
$${\sigma^2}_{mov}==\alpha {\sigma^2}_{mov} + (1-\alpha){\sigma^2}_\phi$$
 
Here $\alpha$ is the “momentum” given to previous moving statistic, around 0.99, and those with $\phi$ subscript are mini-batch mean and mini-batch variance.

### Where to Insert Batch Norm Layers?

Batch normalization may be used on the inputs to the layer before or after the activation function in the previous layer. It may be more appropriate after the activation function if for $S$-shaped functions like the hyperbolic tangent and sigmoid function. 

It may be appropriate before the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types, as the authors of the original paper puts: ‘The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution’.

### Advantages

1. Reduces internal covariate shift
1. Improves the gradient flow
1. More tolerant to saturating nonlinearities because it is all about the range of values fed into those activation functions.
1. Reduces the dependence of gradients on the scale of the parameters or their initialization (less sensitive to weight initialization)
1. Allows higher learning rates because of less danger of exploding/vanishing gradients.
acts as a form of regularization
1. All in all accelerates neural network convergence


### Don’t Use With Dropout
Batch normalization offers some regularization effect, reducing generalization error, perhaps no longer requiring the use of dropout for regularization. Further, it may not be a good idea to use batch normalization and dropout in the same network. The reason is that the statistics used to normalize the activations of the prior layer may become noisy given the random dropping out of nodes during the dropout procedure.



### Jaccard Index and Dice-coefficient
Also called as Intersection over Union. In case of segmentation task, where we want to assign a class to each pixel of the input image. Most likely, there exists huge class imbalance (i.e.) background pixels significantly greater than foreground pixels. In such cases accuracy is not a right metric. However, it's clearly not a good classifier although it might get $90\%$ accuracy. We would ideally want like a metric that is not dependent on the distribution of the classes in the dataset. For this reason, the most commonly used metric for segmentation is Intersection over Union (IoU) and Dice co-efficient (DSC). 
$$IOU(A,B) = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A\cap B}{A \cup B}$$
$$DSC(A,B) = 2 \times \frac{A \cap B}{A + B}$$


## Effect of Hyperparameters

### Batch Size

1. It is well known that too large batch size will lead to poor generalization. On the one extreme, using a batch equal to the entire dataset guarantees convergence to the global optima of the objective function. However, this is at the cost of slower, empirical convergence to that optima.
1.  On the other hand, using smaller batch sizes have been empirically shown to have faster convergence to “good” solutions. This is intuitively explained by the fact that smaller batch sizes allow the model to “start learning before having to see all the data.” 
1. The downside of using a smaller batch size is that the model is not guaranteed to converge to the global optima. It will bounce around the global optima, staying outside some $\epsilon$-ball of the optima where $\epsilon$ depends on the ratio of the batch size to the dataset size.
1. We know a batch size of $1$ usually works quite poorly. It is generally accepted that there is some “sweet spot” for batch size between $1$ and the entire training dataset that will provide the best generalization. This “sweet spot” usually depends on the dataset and the model at question.

