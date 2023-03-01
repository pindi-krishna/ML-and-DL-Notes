# Logistic Regression(LR)

[Stanford notes](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf)

## Introduction

1. This is used for classification.
1. This is same as Linear regression, but as we need to classify, we need discrete outputs not continuous. Therefore, we use sigmoid function at the end.  $$h_\theta(x) = \text{Sigmoid}(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
1.  Here sigmoid gives the probability of that training example belonging to **class 1**. 


## Minimization of Loss Function

1. We want to choose $\theta$ so as to minimize $J(\theta)$.
1. To minimize $J$, we find out its derivative and surprisingly, derivative will end up as same as in case of linear regression. 
1. Let’s consider the gradient descent
algorithm, which starts with some initial $\theta$, and repeatedly performs the update: $$ \theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$
1. If different features have different scales, then the loss is dominated by large scale features. 



## Probabilistic Interpretation

1.  Here in case of binary classification, the outputs have the bernoulli distribution ($0$ and $1$) and in case of multi class classification,  they have multinouli distribution. 
1.  Binary classification :  $$P(y = 1 | x; θ) = h_\theta(x)$$ $$P(y = 0 | x; θ) = 1 - h_\theta(x)$$ $$p(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$
1. Assuming that the $m$ training examples were generated independently, we can then write down the likelihood of the parameters as $$L(\theta) = p(~y | X; \theta)=\Pi_{i=1}^{^m} p(y(i) | x(i); \theta)$$ We try to maximize the log-likelihood function. 
In this process, we end up finding the Cross Entropy loss function.