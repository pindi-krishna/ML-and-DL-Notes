# Neural Networks

# Introduction
#### Neuron
Single neuron == linear regression without applying activation(perceptron). Followed by an activation function gives rise to linear/logistic regression. 

### Neural Network

Simple Neural Network

Each Input will be connected to the hidden layer and the NN will decide the connections. As the dataset size increases, NN performance increases.  

![NN](./Images/NN.jpg)

Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

![NNR](./Images/NNR.png)

In above figure ,Let $n_H$ = \# Hidden neurons
$n_X$ = \# input features, $n_y$ = \#output Classes.
1. Variables :
    
    1. $W_1 : (n_H, n_x) $, $b_1 : (n_H, 1) $
    1. $Z_1 : (n_H, 1) $, $a_1 : (n_H, 1) $
    1. $W_2 : (n_y, n_H)$, $b_2 : (n_y, 1) $
    1. $Z_2 : (n_y, 1) $,$a_2 : (n_y, 1) $
    
1. Forward Propagation : 
    
    1. $Z_1 = W_1 X + b_1 $
    1. $a_1 = g_1(Z_1)$
    1. $Z_2 = W_2 a_1 + b_2$
    1. $a_2 = Sigmoid(Z_2)$
    
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
    
    1. n[l] is the number of neurons in a specific layer l. n[0] denotes the number of neurons input layer. n[L] denotes  the number of neurons in output layer.
    1. g[l] is the activation function in $l$ layer.
    $a[l] = g[l](z[l])$.
    1. w[l] weights is used for z[l], shape : (n[l], n[l-1])
    and b[l] bias for layer l, shape : (n[l],1)
    1. x = a[0], a[L] = y' 
    
    
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
    Dont forget axis=1, keepdims=True
    $$ Output = dA[l-1] = {w[l]}^T. dZ[l]$$ 
    
1. Algorithm :
    
    1. Initialize weights randomly
    1. Forward Propagation
    1. Compute Cost function.
    1. Compute gradients using Back Propagation
    1. Update all weights 
    1. Repeat steps 1-5 till the weights converge.
    
1. Why deep NN works well ?
    
    1. Deep NN makes relations with data from simpler to complex. 1. In each layer it tries to make a relation with the previous layer. E.g.:
    Face recognition application:
    $$Image \Rightarrow Edges \Rightarrow Face parts \Rightarrow Faces \Rightarrow desired face$$
    Audio recognition application:
    $$Audio \Rightarrow Low level sound features like (sss,bb) \Rightarrow Phonemes \Rightarrow Words \Rightarrow Sentences$$
    1. Neural Researchers think that deep neural networks "think" like brains (simple ==> complex)
    
1. Parameters : W and b
1. Hyper parameters (parameters that control the algorithm) are like:
       
    1. Learning rate.
    1. Batch size
    1. Number of iteration.
    1. Number of hidden layers L.
    1. Number of hidden units n.
    1. Choice of activation functions.
    
    You have to try values yourself of hyper parameters.

### Vanishing / Exploding gradients

1.  The Vanishing / Exploding gradients occurs when your derivatives become very small or very big.
1. Illustration : 
Suppose  that we have a deep neural network with number of layers L, and all the activation functions are linear and each b = 0. Then  
    $$ Y' = W[L]W[L-1].....W[2]W[1]X$$ 
1. Exploding Gradients :  If W > I (Identity matrix) the activation and gradients will explode.
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
which will be very large [0  1.5]
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
which will be very small [0  1.5]
    
1. Recently Microsoft trained 152 layers (ResNet)! which is a really big number. With such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient descent will take tiny little steps. It will take a long time for gradient descent to learn anything.
1. There is a partial solution that doesn't completely solve this problem but it helps a lot - careful choice of how you initialize the weights.



# Optimization
### Stochastic gradient descent:
#### Why
Batch gradient descent requires huge computation power when dealing with millions of samples because to make one update entire dataset gradients should be added.
#### How
To reduce the computation power and time, we do the update for every datapoint. 
#### Problem
Too noisy regarding cost minimization because each point may updates/moves the parameters in a different direction. That is why, we need a method which is not noisy and requires less computation time and power. 

### Mini-batch gradient descent

1. Here, we do the update for every batch of data points. Usually batch size is less than m and greater than 1. 

1. Mini batch size is between 1 and m. It has to be a power of 2 (because of the way computer memory is layed out and accessed, sometimes your code runs faster if your mini-batch size is a power of 2): 64, 128, 256, 512, 1024, ...
1. Mini-batch gradient descent works much faster in case of large datasets.


#### Problem with Gradient descent
Wherever the error surface is steep, the loss decreases very quickly and wherever it is gentle, the loss decreases very slowly. To move on the gentle surface, usually it requires a lot of updates to be done. This increases the time complexity. Therefore, we need a method which moves very quickly on the gentle error surface also. 

### Gradient descent with momentum

#### Intuition

1. When moving on the error surface, we not only consider the gradient direction at that particular timestep but also consider the history i.e the gradient direction is before timesteps. 
1. The gradient of the cost function at saddle points( plateau) is negligible or zero, which in turn leads to small or no weight updates. Hence, the network becomes stagnant, and learning stops.
1. If gradient direction at the previous timesteps and present timestep is same which is usually the case when moving on gentle surface, then we move by large amount. This is how, we can escape the gentle surface part quickly. 

1. Usually we give less importance to the gradient at the present time step and give more importance to history(gradients of past timesteps). 
1. It also smoothens out the averages of skewed data points (oscillations w.r.t. Gradient descent terminology). So this reduces oscillations in gradient descent and hence makes faster and smoother path towards minima.


#### Exponentially weighted averages

1. General equation :
 $$V_t = \beta \times V_{t-1} + (1-\beta) \times \nabla W_t$$
 given, $V_o = 0$
1. If we plot this it will represent averages over $\approx \frac{1}{(1 - \beta)}$ entries:
    
    1. $\beta = 0.9$ will average last 10 entries
    1. $\beta = 0.98$ will average last 50 entries
    1. $\beta = 0.5$ will average last 2 entries
    
Best $\beta$ average for our case is between 0.9 and 0.98.
1. Here we are giving more weight to the previous derivatives than the derivative of the current batch so that if the current batch has a lot of noise or turns out to contain more number of outliers, then all that noise will be given a less weightage. 

1. The momentum algorithm almost always works faster than standard gradient descent. The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights with the new values which will also reduce the noise. 


#### Update rule
$$V_t = \beta \times V_{t-1} + (1 - \beta) \times \nabla W_t$$
$$W_{t+1} = W_t - \eta \times V_{t+1}$$
given, $V_o = 0$ and $\eta$ - learning rate.  
#### Limitation


1. In GD with momentum, the update rule is  
$$W_{t+1} = W_t - \underset{1^{st} move}{\eta \times \beta \times V_{t-1}} - \underset{2^{nd} move}{\eta \times (1-\beta)\nabla W_t}$$
While moving on the error surface, If the  gradient direction in the previous timesteps is similar as the present timesteps, then gradient picks up the velocity, then it take large steps and overshoots. Then take u-turn, overshoot and again take u-turn, overshoot and this may go on for a while. 
1. Also, It may overshoot and enter another valley with local minima, where the gradient direction can completely be  changed and tries to converge at the local minima which shouldn’t be the case. 


### Nesterov accelerated gradient descent (NAG)
#### Intuition

1. In GD with momentum, the first move is due to the history and 2nd move is due to the gradient of W. So we know, that we are going to move by atleast by $\gamma.v_{t}$ and then a bit more by $\eta.\nabla W$. Then why not compute the gradient $\nabla W_{look-ahead}$ at this paritally updated value of W ($\nabla W_{look-ahead} = W_t - \gamma.V_t$) instead of computing it using the current value $w_t$. Then do the update based on $\nabla W_{look-ahead}$.
1. Looking ahead helps NAG in correcting its course quicker than momentum based gradient descent. 


#### Update rule
$$W_{look-ahead} = W_t - \gamma.V_{t-1}$$
$$V_t = \gamma.V_{t-1} + \eta.\nabla W_{look-ahead}$$
$$W_{t+1} = W_t - V_t$$


### AdaGrad optimization algorithm
#### Intuition

1. In the real-world dataset, some features are sparse (for example, in Bag of Words most of the features are zero so it’s sparse) and some are dense (most of the features will be noon-zero), so keeping the same value of learning rate for all the weights is not good for optimization. Some features gets updated many times and other features get updated fewer number of times.  
1. Hence, In case of sparse feature, this is imporatant and cant be ignored, then we would want to take the updates of that particular feature also very seriously. 
1. Therefore, can we have a different learning rate for each parameter which takes care of the frequency of features. 

#### Update rule
Decay the learning rate for parameters in proportion to their update history. 
$$V_t = V_{t-1} + {(\nabla W_t)}^2$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t + \epsilon}} \times \nabla W_t$$
If any feature get too many updates, then learning rate of that feature is decreased and vice versa. 
#### Observation
In practice, this does not work so well if we remove the sqaure root from the denominator. (No idea why this happens).

#### Limitation
Over time the effective learning rate for b will decay to an extent that there will be no further updates to b (in the direction of b). 
Adagrad decays the LR very aggressively (as the denominator grows), as a result after a while the frequent parameters will start receiving very small updates because of the decayed LR.
### Root mean square prop(RMS Prop)
#### Intuition

1. The basic idea is that if there is one parameter in the neural network that makes the estimate of the cost function J oscillate a lot, you want to penalize the update of this parameter during optimization, so to avoid the gradient descent algorithm adapt too quickly to changes in this parameter, as compared to the others.
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
These are used to convert the linear input signals of a node into non-linear output signals to facilitate the learning of high order polynomials that go beyond one degree for deep networks. 

### Sigmoid
#### Introduction

  $$ Sigmoid(x) = \frac{1}{1 + e^{-x}}$$
Sigmoid activation function range is [0,1]. Therefore, If the classification is between 0 and 1, Then use the output activation as sigmoid.

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
1. If your unlucky a neuron may be never active because the initialization has put it outside the manifold.
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
where $\alpha$ is a very small number (usually 0.01) which can be a
if(x > 0) hyperparameter or learned through. 

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
Yes, if the activation function of the network is not zero centered, $y=f(x^T w)$ is always positive or always negative. Thus, the output of a layer is always being moved to either the positive values or the negative values. As a result, the weight vector needs more updates to be trained properly, and the number of epochs needed for the network to get trained also increases. This is why the zero centered property is important, though it is NOT necessary.
\par
Zero-centered activation functions ensure that the mean activation value is around zero. This property is important in deep learning because it has been empirically shown that models operating on normalized data––whether it be inputs or latent activations––enjoy faster convergence.
\par
Unfortunately, zero-centered activation functions like tanh saturate at their asymptotes –– the gradients within this region get vanishingly smaller over time, leading to a weak training signal.
\par
ReLU avoids this problem but it is not zero-centered. Therefore all-positive or all-negative activation functions whether sigmoid or ReLU can be difficult for gradient-based optimization. So, To solve this problem deep learning practitioners have invented a myriad of Normalization layers (batch norm, layer norm, weight norm, etc.). we can normalize the data in advance to be zero-centered as in batch/layer normalization.

# Parameters Initialization for Deep Networks
Justification is Pending for all different initializations.
#### WHY ?

1.  Initialization of parameters, if done correctly then optimization will be achieved in the least time otherwise converging to a minima using gradient descent will be impossible.


### Zero/Same Initialization
#### Introduction

1. Initializing all parameters equally (zero or another value).
1.  Biases have no effect what so ever when initialized with 0.

#### Limitation
Initializing all the weights equally leads the neurons to learn the same features during training. This causes symmetry breaking problem which means that if two hidden units are connected to the same input units, then these should have different initialization or else the gradient would update both the units in the same way and we don't learn anything new by using an additional unit. The idea of having each unit learn something different motivates random initialization of weights which is also computationally cheaper.
### Random Initialization
#### Intuition

1. Biases are often chosen heuristically (zero mostly) and only the weights are randomly initialized, almost always from a Gaussian or uniform distribution. 
1. The scale of the distribution is of utmost concern. Large weights might have better symmetry-breaking effect but might lead to chaos (extreme sensitivity to small perturbations in the input) and exploding values during forward & back propagation. 
1. As an example of how large weights might lead to chaos, consider that there's a slight noise adding $\epsilon$ to the input. Now, we if did just a simple linear transformation like W * x, the $\epsilon$ noise would add a factor of $W \times \epsilon$ to the output. In case the weights are high, this ends up making a significant contribution to the output.
1. If the weights are initialized with a to high value the activations will explode and saturate to 1. If the weights are initiliazed to small values, the activations will vanish and saturate to zero. 


#### Uniform Distribution

1.  Initialize weights from a uniform distribution with n as the number of inputs in $l^{th}$ layer. 
    $$W_{i,j} \approx UniformDistribution(-\frac{1}{\sqrt{n^{l-1}}},\frac{1}{\sqrt{n^{l-1}}})$$


### Xavier Initialization
#### Xavier Uniform

1.  Initialize weights from a uniform distribution. 
    $$W_{i,j} \approx UniformDistribution(-\frac{\sqrt{6}}{\sqrt{n^{l-1} + n^l}},\frac{\sqrt{6}}{\sqrt{n^{l-1} + n^l}})$$
1. Works well for sigmoid


#### Xavier Normal

The goal of Xavier Initialization is to initialize the weights such that the variance of the activations are the same across every layer. This constant variance helps prevent the gradient from exploding or vanishing.
$$W_{ij} \approx  \mathcal{N}(0, \sigma^2)$$
$$\sigma^2 = \frac{2}{\sqrt{n^{l-1} + n^l}}$$
1. Assumptions : 

    1. Weights and inputs are centered at zero.
    1. Weights and inputs are independent and identically distributed.
    1. Biases are initialized as zeros.
    1. We use the tanh activation function, which is approximately linear with small inputs, (i,e) $Var(a^l) = Var(z^l)$

1. Xavier initialization is designed to work well with tanh or sigmoid activation functions.


### HE Initialization
#### HE Uniform

1.  Initialize weights from a uniform distribution. 
    $$W_{i,j} \approx UniformDistribution(-\frac{\sqrt{6}}{\sqrt{n^{l-1}}},\frac{\sqrt{6}}{\sqrt{n^{l-1}}})$$
1. Works well for sigmoid


#### HE Normal

1. Works well with ReLU.
$$W_{ij} \approx  \mathcal{N}(0, \sigma^2)$$
$$\sigma^2 = \frac{2}{\sqrt{n^{l-1}}}$$


# Regularization

### Introduction

1. The idea here is to limit the capacity (the space of all possible model families) of the model by adding a parameter norm penalty,$\Omega(\theta)$, to the objective function J. 
$$\hat{J(\theta;X,y)}=J(\theta;X,y)+\lambda \Omega(\theta)$$
Here, $\theta$ represents only the weights and not the biases, the reason being that the biases require much less data to fit and do not add much variance.
1. We usually do not regularize the bias, since it generally has a lower variance than the weights, due to the bias not interacting with both the inputs and the outputs, as the weights do.
1. For example for a linear model $f(x)=mx+b$ the bias term allows us to fit lines that do not pass through the origin - however if we do weight deacy on the bias term, we are encouraging the bias to stay close to 0, which defeats the purpose of the bias term in the first place.


#### Basic Recipe for reducing bias and variance

1. If your algorithm has a high bias :
    
    1. Try to make your NN bigger (size of hidden units, number of layers)
    1. Try a different model that is suitable for your data.
    1. Try to run it longer.
    1. Different (advanced) optimization algorithms.
    
1. If your algorithm has a high variance :
    
    1. More data.
    1. Try regularization.
    1. Try decreasing the complexity of the model.
    


### L1 and L2 Regularization

1. L1 matrix norm is sum of absolute values of all w.
$$||W|| = \sum |w[i,j]|$$ 

The L1 regularization Loss:
$$J(w,b) =  \frac{1}{m} \times \sum (L(y(i),y'(i))) + (\frac{\lambda}{2m}) \times \sum ||W[l]||$$
1. L2 matrix norm also called as Frobenius norm is sum of all w squared.
$$||W||^2 = Sum(|w[i,j]|^2) = W^T.W $$

The L2 regularization Loss:
$$J(w,b) =  \frac{1}{m} \times \sum (L(y(i),y'(i))) + (\frac{\lambda}{2m}) \times \sum ||W[l]||^2$$

1. In practice this penalizes large weights and effectively limits the freedom in your model.
1. The new term $(1 - \frac{\alpha \times \lambda}{m}) \times w[l]$ causes the weight to decay in proportion to its size.
1. L2 regularization retains the weights corresponding to features with high eigen values and scales down significantly the weights of the features with low eigen values. (This is what usually happens in PCA. Features with low eigen values in the covariance vector are removed). L2 regularization proof can be found in mithesh khapra deep learning lecture 62. 
1. In most cases, L2 regularization is being used. L2 penalizes the outliers more than L1. L1 can make unimportant feature weights zero.


#### Why regularization reduces overfitting ? 

Here are some intuitions:

1. In general, whenever we train a neural network, we give a lot of freedom to weights. so, weights can take any real value and try to make the training error zero (overfitting). Therefore, we try to restrict the weights to between certain range, or penalize its magnitude so that it doesn't overfit the training data and generalize well on the test data. 

1. If lambda is too large - a lot of w's will be close to zeros which will make the NN simpler (you can think of it as it would behave closer to logistic regression).
If lambda is good enough it will just reduce some weights that makes the neural network overfit.
1. L1 regularizer can make the weights zero which can also used for feature selection.  


### Data augmentation


1. Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.
1. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.
1. For a classification task, we desire for the model to be invariant to certain types of transformations, and we can generate the corresponding (x,y) pairs by translating the input x. But for certain problems, like density estimation, we can't apply this directly unless we have already solved the density estimation problem.

1. However, caution needs to be mentioned while data augmentation to make sure that the class doesn't change. For e.g., if the labels contain both "b" and "d", then horizontal flipping would be a bad idea for data augmentation. Add random noise to the inputs is another form of data augmentation, while adding noise to hidden units can be seen as doing data augmentation at multiple levels of abstraction.

1. Finally, when comparing machine learning models, we need to evaluate them using the same hand-designed data augmentation schemes or else it might happen that algorithm A outperforms algorithm B, just because it was trained on a dataset which had more / better data augmentation.


### Noise Robustness
#### Adding Noise to the input

1. Usually, we corrupt the original image/input by adding gaussian noise (sampled from gaussian distribution). Similar to data augmentation. 

1. Adding gaussian noise to the input works similar to L2 regularization. Dimension specific scaling of weights will happen which scales down the non-important weights significantly.  [Mathematical proof can be found](https://www.youtube.com/watch?v=agGUR06jM_g&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU&index=65).

$$\hat{x_i} = x_i + \epsilon$$
$$\hat{y} = \sum w_i x_i$$
$$\tilde{y} = \sum w_i \hat{x_i}$$

$\epsilon$ -- small values drawn from gaussian distribution. 
Usually, our goal is to minimze the squared error $E(\tilde{y} - y)^2$. If we substitute $\tilde{y}$ in the squared error equation and try to get $E(\hat{y} - y)^2$ outside, then equation will end up in the form of regularization.
$$\hat{J(\theta;X,y)}=J(\theta;X,y)+\lambda \Omega(\theta)$$


#### Adding noise to weights
Noise can even be added to the weights. This has several interpretations. One of them is that adding noise to weights is a stochastic implementation of Bayesian inference over the weights, where the weights are considered to be uncertain, with the uncertainty being modelled by a probability distribution. It is also interpreted as a more traditional form of regularization by ensuring stability in learning.

#### Label Smoothing
Label Smoothing is a regularization technique that introduces noise for the labels. This accounts for the fact that datasets may have mistakes in them, so maximizing the likelihood of 
$log p(y|x)$ directly can be harmful. Assume for a small constant $\epsilon$, the training set label y is correct with probability $1 - \epsilon$ and incorrect otherwise. Label Smoothing regularizes a model based on a softmax with $k$ output values by replacing the hard 0 and 1 classification targets with targets of $\frac{\epsilon}{k-1}$ and $1 - \epsilon$ respectively.

### Early stopping :


1. In this technique we plot the training set and the validation set cost together for each iteration. At some iteration the validation set cost will stop decreasing and will start increasing.

1. We will pick the point at which the training set error and validation set error are best (lowest training cost with lowest validation cost). We will take these parameters as the best parameters.

1. However, since we are setting aside some part of the training data for validation, we are not using the complete training set. So, once Early Stopping is done, a second phase of training can be done where the complete training set is used. There are two choices here:

1. Train from scratch for the same number of steps as in the Early Stopping case.
1. Use the weights learned from the first phase of training and retrain using the complete data.


1. Other than lowering the number of training steps, it reduces the computational cost also by regularizing the model without having to add additional penalty terms. 

1. It allows only t updates to the parameters. It affects the optimization procedure by restricting it to a smal volume of the parameter space, in the neighbourhood of the initial parameters ($\theta$).

1. Early stopping is another way of regularization. But, it also can be used along with regularization (L2 or L1).

1. Under certain assumptions, this is similar to L2 regularization. [Mathematical Proof can be found here](https://www.youtube.com/watch?v=zm5cqvfKO-o&list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU&index=67).


### Parameter Tying and Parameter Sharing
Till now, most of the methods focused on bringing the weights to a fixed point. However, there might be situations where we might have some prior knowledge on the kind of dependencies that the model should encode. Suppose, two models A and B, perform a classification task on similar input and output distributions. In such a case, we'd expect the parameters ($W_a$ and $W_b$) to be similar to each other as well. We could impose a norm penalty on the distance between the weights, but a more popular method is to force the set of parameters to be equal. This is the essence behind Parameter Sharing. A major benefit here is that we need to store only a subset of the parameters (e.g. storing only $W_a$ instead of both $W_a$ and $W_b$) which leads to large memory savings.

### Bagging

#### Introduction
The techniques which train multiple models and take the maximum vote across those models for the final prediction are called ensemble methods. The idea is that it's highly unlikely that multiple models would make the same error in the test set.

#### Algorithm

1. Train multiple independent models.
1. At test time average their results.

#### Limitation
Computationally very expensive while both training and testing.

### Dropout Regularization

#### Why ?
The problem with bagging is that we can't train an exponentially large number of models and store them for prediction later. Dropout makes bagging practical by making an inexpensive approximation. In a simplistic view, dropout trains the ensemble of all sub-networks formed by randomly removing a few non-output units by multiplying their outputs by 0.

#### Introduction

1. Dropout refers to dropping out units/neurons. Temporarily remove a node and all its incoming/outgoing connections resulting in a thinned network. 
1. Each node/neuron is retained with a fixed probability (usually 0.5) for hidden nodes and p = 0.8 for visible nodes. 
1. Tricks: Share the wrights across all the networks and Samples a different network for each training instance. 
1. If there are n neurons, then $2^n$ networks are possible to train. Each thinned network gets trained rarely (or even never) but the parameter sharing ensures that no model has untrained or poorly trained parameters.
1. At test time, it is impossible to aggregate the output of $2^n$ thinned networks. Instead we use the full neural network and scale the output of each node by the fraction of times it was on during training. 
1. Dropout prevents hidden units form co-adapting. Essentially a hidden cannot rely too much on other units as they may get dropped out any time. Each hidden unit has to learn to be more robust to these random dropouts. 

#### Algorithm
In case of dropout, The forward propagation becomes 
$$Input  = A[l-1]$$
$$ Z[l] = {W[l]}^T.A[l-1] + b[l]$$
$$ A[l] = g[l](Z[l])$$
$$ Output =  D[l] (A[l])$$

Where D is the dropout layer (Masking the activations $D[l] \in {0,1}$). The key factor in the dropout layer is $keep\_prob$ parameter, which specifies the probability of keeping each unit. Say if $keep\_prob = 0.8$, we would have $80\%$ chance of keeping each output unit as it is, and $20\%$ chance set them to 0. 

Code snippet would be like: 

    keep_prob = keep_probs[i-1]
    D = np.random.rand(A.shape[0], A.shape[1])
    D = ($D < keep_prob$).astype(int)
    A = np.multiply(D, A) 

#### Differences between bagging and Dropout

1. In bagging, the models are independent of each other, whereas in dropout, the different models share parameters, with each model taking as input, a sample of the total parameters.
1. In bagging, each model is trained till convergence, but in dropout, each model is trained for just one step and the parameter sharing makes sure that subsequent updates ensure better predictions in the future


#### Intuition
Adding noise in the hidden layer is more effective than adding noise in the input layer. For e.g. if some unit $h_i$ learns to detect a nose in a face recognition task. Now, if this $h_i$ is removed, then some other unit either learns to redundantly detect a nose or associates some other feature (like mouth) for recognising a face. In either way, the model learns to make more use of the information in the input. On the other hand, adding noise to the input won't completely removed the nose information, unless the noise is so large as to remove most of the information from the input.



#### Dropout at Test time
We don't turn off any neurons during test time. We consider all the neurons but only probability as each and every neuron is active only with probability p, during training time. 
We multiply its activation value only with probablity 'p' to get the predictions.


# Auto Encoders

### Introduction
#### Latent Vector or Thought Vector
Neural networks have the rather uncanny knack for turning meaning into numbers. Data flows from the input to the output, getting pushed through a series of transformations which process the data into increasingly abstruse vectors of representations. These numbers, the activations of the network, carry useful information from one layer of the network to the next, and are believed to represent the data at different layers of abstraction.
Thought vectors have been observed empirically to possess some curious properties. 

![latent_smile](./Images/Latent_Smile.png)

#### The encoder of a linear autoencoder is equivalent to PCA if:

1. Use a linear encoder.
1. Use a linear decoder.
1. Use mean squared error loss function. 
1. Standardize the inputs. 


### Regularization in auto-encoders

#### Tieing weights
This is one of the trick to reduce over-fitting. We enforce $W^* = W^T$. Here, we have only one W to update. 
During the back propagation, w.r.t to decoder, we find $\partial L / \partial W$ as we usually do. However, w.r.t to encoder, we add derivative computed w.r.t to decoder to derivative w.r.t encoder to find the final derivative w.r.t encoder.  

### Denoising auto-encoders

We corrupt the data with some low probability which usually we encounter in the test data and pass that corrupted input to the auto-encoder and try to reconstruct the original data. 

#### Intuition

1. The model gets robust to the expected noise in the test set and perform well on test set too. 
1. For ex, in case of BMI prediction using height, weight parameters, if we corrupt height, then to reconstruct the original height (without corruption), the AE model should learn the interactions between height and weights so that it can figure out what went wrong and correct the corruption. 

Empirically, it has been found that, In case of gausssian noise added AE filters captures more meaningful patterns like edges, corners than simple L2 regularized AE. 

### Sparse Auto-encoders

We use a different type of regularization technique to restrict the freedom the weights. In sparse AE, we ensure that the neuron is inactive most of the times. 

The constraint imposed is on the average value of the activation of neurons. The average activation of neurons in layer $l$ is given by 
$$\hat{\rho} = \frac{1}{k}\sum_i^k {h(x_i)}_l$$
where $k$ is the number of the neurons in the layer l. 
We are trying to keep the average value of the activations of neurons to be close to small value ($\rho == 0.005$).

#### Intuition: 

We are trying to prevent the neuron from firing most of the times by enforcing the above constraint . As the neuron fires very few number of times, it tries to learn meaningful patterns during training to reduce the training error. 

#### How ? 
$$\omega(\theta) = \sum_{i=1}^k \rho log\frac{\rho}{\hat{\rho}} + (1-\rho) log\frac{1- \rho}{1 - \hat{\rho}}$$

### Contractive auto-encoders

#### Intuition

$$Total Loss = L(\theta) + \Omega(\theta)$$

1. $L(\theta)$ -- Prediction error. Minimizing this will capture important variations in data. 
2. $\Omega(\theta)$ -- Regularization error. Minimizing this do not capture variations in data. 
3. Tradeoff -- Captures only very important variations in data. 


# Practical Aspects of Deep Learning

### Train/Validation/Test sets

1.  Its impossible to get all your hyperparameters right on a new application from the first time. So the idea is you go through the loop: Idea ==> Code ==> Experiment. You have to go through the loop many times to figure out your hyperparameters.
1. Your data will be split into three parts:
a) Training set. (Has to be the largest set)
b) Hold-out cross validation set / Development or "dev" set.
c) Testing set.
1. You will try to build a model upon training set then try to optimize hyperparameters on validation set as much as possible. Then after your model is ready you try and evaluate the testing set.
1.  So, the general trend on the ratio of splitting the models:
    
    1. If size of the dataset is 100 to 1000000 ==> 60/20/20
    1.  If size of the dataset is 1000000 to INF ==> 98/1/1 or 99.5/0.25/0.25
    

1. Make sure the validation and test set are coming from the same distribution.
For example, if cat training pictures is from the web and the validation/test pictures are from users cell phone, then they will mismatch. It is better to make sure that validation and test set are from the same distribution.
1.  The training error will generally increase as you increase the dataset size. This is because the model will find it harder to fit to all the datapoints exactly now. Also, by increasing the dataset size, your validation (dev) error will decrease as the model would learn to be more generalized now


### Normalizing inputs

1. Why normalize?

    1. If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then optimizing it will take a long time.
    1. But if we normalize it, the opposite will occur. The shape of the cost function will be consistent (look more symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be faster.

1. Normalizing the inputs will speed up the training process a lot.


### Shuffling the dataset

#### Should we shuffle the dataset before training ?

The obvious case where you'd shuffle your data is if your data is sorted by their class/target. Here, you will want to shuffle to make sure that your training/test/validation sets are representative of the overall distribution of the data. The idea behind batch gradient descent is that by calculating the gradient on a single batch, you will usually get a fairly good estimate of the "true" gradient. That way, you save computation time by not having to calculate the "true" gradient over the entire dataset every time.

#### Should you shuffle the dataset after every epoch ?

You want to shuffle your data after each epoch because you will always have the risk to create batches that are not representative of the overall dataset, and therefore, your estimate of the gradient will be off. Shuffling your data after each epoch ensures that you will not be "stuck" with too many bad batches.
\par
In regular stochastic gradient descent, when each batch has size 1, you still want to shuffle your data after each epoch to keep your learning general. Indeed, if data point 17 is always used after data point 16, its own gradient will be biased with whatever updates data point 16 is making on the model. By shuffling your data, you ensure that each data point creates an "independent" change on the model, without being biased by the same points before them.

### Batch Normalization

[Source](https://mmuratarat.github.io/2019-03-09/batch-normalization)

#### Why

Typically, we train model in mini-batches. Initially, we make sure the input layer is zero centered and unit gaussian. But in the deep network, there is a possibility that the distributions of hidden layers may change which makes the learning process hard. That's why, Why not ensure that the pre-activations at each layer have similar distributions which makes the learninig easy. 

#### Introduction

Batch Normalization will let the network decide what is the best distributions for it instead of forcing it to have zero mean and unit variance. This is usually applied with mini-batches. This is a co-variate shift. Training the model on different possible distributions of data. 

#### Algorithm

Note that, all the expressions above implicitly assume broadcasting as X is of size (N,D) and both $\mu_\phi$ and ${\sigma^2}_\phi$ have size equal to (D).

$$\hat{x_{il}}=\frac{x_{il} - \mu_{\phi l}}{\sqrt{{\sigma^2}_{\phi l} + \epsilon}}$$
where
$$\mu_{\phi l}=\frac{1}{N} \sum_{p=1}^N x_{pl}$$
and
$${\sigma^2}_{\phi l}=\sum_{p=1}^N \frac{1}{N} (x_{pl} - \mu_{\phi l})^2$$
with i=1…,N and l=1,…,D. 

**The pairs ($\gamma^k$, $\beta^k$) are learnt per neuron/dimension**

#### Batch normalization at test time

In testing we might need to process examples one at a time. The mean and the variance computed for training batches won't make sense to use in case of of one test sample. We have to compute an estimated value of mean and variance to use it in testing time. 

Exponentially weighted “moving average” can be used as global statistics to update population mean and variance:
$$\mu_{mov} = \alpha \mu_{mov}+(1-\alpha)\mu_\phi$$
$${\sigma^2}_{mov}==\alpha {\sigma^2}_{mov} + (1-\alpha){\sigma^2}_\phi$$
 
Here $\alpha$ is the “momentum” given to previous moving statistic, around 0.99, and those with $\phi$ subscript are mini-batch mean and mini-batch variance.

#### Where to Insert Batch Norm Layers?

Batch normalization may be used on the inputs to the layer before or after the activation function in the previous layer. It may be more appropriate after the activation function if for s-shaped functions like the hyperbolic tangent and logistic function. 

It may be appropriate before the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types, as the authors of the original paper puts: ‘The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution’.

#### Advantages

1. Reduces internal covariate shift
1. Improves the gradient flow
1. More tolerant to saturating nonlinearities because it is all about the range of values fed into those activation functions.
1. Reduces the dependence of gradients on the scale of the parameters or their initialization (less sensitive to weight initialization)
1. Allows higher learning rates because of less danger of exploding/vanishing gradients.
acts as a form of regularization
1. All in all accelerates neural network convergence


#### Don’t Use With Dropout
Batch normalization offers some regularization effect, reducing generalization error, perhaps no longer requiring the use of dropout for regularization. Further, it may not be a good idea to use batch normalization and dropout in the same network. The reason is that the statistics used to normalize the activations of the prior layer may become noisy given the random dropping out of nodes during the dropout procedure.

### Metrics

#### Jaccard Index and Dice-efficient
Also called as Intersection over Union. In case of segmentation task, where we want to assign a class to each pixel of the input image. Most likely, there exists huge class imbalance i,e background pixels significantly greater than foreground pixels. In such cases accuracy is not a right metric.However, it's clearly not a good classifier although it might get $90\%$ accuracy. We would ideally want like a metric that is not dependent on the distribution of the classes in the dataset. For this reason, the most commonly used metric for segmentation is Intersection over Union (IoU) and Dice co-efficient (DSC). 
$$IOU(A,B) = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A\cap B}{A \cup B}$$
$$DSC(A,B) = \frac{2 A \cap B}{A + B}$$

#### Precision

Precision represents the fraction of detections that were actually true, whereas Recall stands for the the fraction of true events that were successfully detected.
$$Precision = \frac{\text{TP}}{\text{TP + FP}}$$
$$Recall = \frac{\text{TP}}{\text{TP + FN}}$$
For example, in a particular sample of population, one out of a 1000 people might have cancer. Thus, 9999 people don't have cancer. If we simply use a classifier, then that classifier predicts everyone as not having cancer, we can achieve an accuracy of $99.99\%$. But would we be willing to use this classifier for testing ourself? Definitely not. In such a case, accuracy is a bad metric. We instead use precision and recall to evaluate our classifier. Now, consider that if a detector says that all the cases are not cancer will achieve the perfect precision, but 0 recall. Many a times, it's actually desirable to have a single metric to judge on, rather than have a trade-off between two of them. F1-score, which is the Harmonic Mean of Precision & Recall is a widely accepted metric:
$$F1-score = 2 \times \frac{\text{precision} \times \text{ recall}}{\text{precision + recall}}$$

However, F1-score gives equal weightage to both precision and recall. There can be cases where you want to weigh one over the other and hence, we have the more general, F-beta score:

$$F\beta -score = (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{\beta^2 \text{precision} + \text{recall}}$$
$\beta$ = 1$\implies$ Recall and Precision has given equal weightage \newline
$\beta$ < 1$\implies$ Precision has given more weightage than Recall \newline
$\beta$ > 1$\implies$ Recall is given more weightage than precision \newline

#### Mean Average Precision (mAP)

Mean Average Precision (mAP) is a popular evaluation metric used for object detection (i.e. localisation and classification). mAP is finding the area under the precision-recall curve above.

### Effect of Hyperparameters

#### Batch Size

1. It is well known that too large batch size will lead to poor generalization. On the one extreme, using a batch equal to the entire dataset guarantees convergence to the global optima of the objective function. However, this is at the cost of slower, empirical convergence to that optima.
1.  On the other hand, using smaller batch sizes have been empirically shown to have faster convergence to “good” solutions. This is intuitively explained by the fact that smaller batch sizes allow the model to “start learning before having to see all the data.” 
1. The downside of using a smaller batch size is that the model is not guaranteed to converge to the global optima. It will bounce around the global optima, staying outside some ϵ-ball of the optima where ϵ depends on the ratio of the batch size to the dataset size.
1. We know a batch size of 1 usually works quite poorly. It is generally accepted that there is some “sweet spot” for batch size between 1 and the entire training dataset that will provide the best generalization. This “sweet spot” usually depends on the dataset and the model at question.

