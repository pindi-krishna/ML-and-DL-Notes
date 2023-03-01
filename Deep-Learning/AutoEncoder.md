# Auto Encoders
[Source](https://gabgoh.github.io/ThoughtVectors/)
# Introduction

## Latent Vector or Thought Vector
Neural networks have the rather uncanny knack for turning meaning into numbers. Data flows from the input to the output, getting pushed through a series of transformations which process the data into increasingly abstruse vectors of representations. These numbers, the activations of the network, carry useful information from one layer of the network to the next, and are believed to represent the data at different layers of abstraction.
Thought vectors have been observed empirically to possess some curious properties. 

![latent_smile](./Images/Latent_Smile.png)

## The encoder of a linear autoencoder is equivalent to PCA if:

1. Use a linear encoder.
1. Use a linear decoder.
1. Use mean squared error loss function. 
1. Standardize the inputs. 


# Regularization in auto-encoders

## Intuition

$$Total Loss = L(\theta) + \Omega(\theta)$$

1. $L(\theta)$ -- Prediction error. Minimizing this will capture important variations in data. 
2. $\Omega(\theta)$ -- Regularization error. Minimizing this do not capture variations in data. 
3. Tradeoff -- Captures only very important variations in data. 

## Tying weights
This is one of the trick to reduce over-fitting. We enforce $W^* = W^T$. Here, we have only one $W$ to update. 
During the back propagation, w.r.t decoder, we find $\partial L / \partial W$ as we usually do. 

**However, we add derivative computed w.r.t decoder to derivative w.r.t encoder to find the final derivative w.r.t encoder.**  

## Denoising auto-encoders

We corrupt the data with some low probability which usually we encounter in the test data and pass that corrupted input to the auto-encoder and try to reconstruct the original data. 

## Intuition

1. The model gets robust to the expected noise in the test set and perform well on test set too. 
1. For ex, in case of BMI prediction using height, weight parameters, if we corrupt height, then to reconstruct the original height (without corruption), the AE model should learn the interactions between height and weights so that it can figure out what went wrong and correct the corruption. 

Empirically, it has been found that, In case of gausssian noise added AE filters captures more meaningful patterns like edges, corners than simple L2 regularized AE. 

## Sparse Auto-encoders

We use a different type of regularization technique to restrict the freedom of weights. In sparse AE, we ensure that the neuron is inactive most of the times. 

The constraint imposed is on the average value of the activation of neurons. The average activation of neurons in layer $l$ is given by 
$$\hat{\rho} = \frac{1}{k}\sum_i^k {h(x_i)}_l$$
where $k$ is the number of the neurons in the layer $l$. 
We are trying to keep the average value of the activations of neurons to be close to small value ($\rho = 0.005$).

## Intuition: 

We are trying to prevent the neuron from firing most of the times by enforcing the above constraint . As the neuron fires very few number of times, it tries to learn meaningful patterns during training to reduce the training error. 

## How ? 
$$\Omega(\theta) = \sum_{i=1}^k \rho log\frac{\rho}{\hat{\rho}} + (1-\rho) log\frac{1- \rho}{1 - \hat{\rho}}$$

## Contractive auto-encoders
[Source](https://iq.opengenus.org/contractive-autoencoder/)
## Introduction
1. Contractive autoencoder simply targets to learn invariant representations to unimportant transformations for the given data.

1. It only learns those transformations that are provided in the given dataset so it makes the encoding process less sensitive to small variations in its training dataset.

1. The goal of Contractive Autoencoder is to reduce the representation’s sensitivity towards the training input data.

## Formula
1. Frobenius norm of the Jacobian matrix of hidden activations. 
    $$\Omega(\theta) = ||{J_x(h)||_F}^2$$
    $$J_x(h) = \sum_{j=1}^n \sum_{l=1}^k {\left(\frac{\partial h_l}{\partial x_j} \right)}^2$$
    $h_l$ represents the latent representation learnt by the last layer of encoder. 
1. If this value is zero, it means that as we change input values, we don't observe any change on the learned hidden representations.

1. But if the value is very large, then the learned representation is unstable as the input values change.
