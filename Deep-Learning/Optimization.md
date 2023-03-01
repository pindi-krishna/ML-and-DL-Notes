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
