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
