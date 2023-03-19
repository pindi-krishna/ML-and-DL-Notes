# Generative Learning algorithms

## Introduction

1.  Algorithms that try to learn $P(y|x)$ directly (such as logistic regression),
or algorithms that try to learn mappings directly from the space of inputs $X$ to the labels $\{0, 1\}$, (such as the perceptron algorithm) are called discriminative learning algorithms.
1.  Algorithms that instead try to model $P(x|y)$ are generative learning algorithms.
1.  For instance, if $y$ indicates whether an example is a dog $(0)$ or an elephant $(1)$, then $P(x|y = 0)$ models the distribution of dogs features, and $P(x|y = 1)$ models the distribution of elephants features.


## Gaussian Discriminant Analysis

1.  In this model, we’ll assume that $P(x|y)$ is distributed according to a multivariate normal distribution.

1.  This Algorithm make use of Bayes theorem to model $P(y|x)$. 
    
    $$ p(x; \mu, \Sigma) = \sqrt{\frac{1}{{(2 \pi)}^n det(\Sigma)}} exp(-\frac{1}{2}(x-\mu^T\Sigma^{-1}(x-\mu))$$ 

    Here, $\Sigma$ = Covariance of $X$. 
    Here, $\mu$ and $\Sigma$ is computed for each class inputs separately.
    $$y \approx Bernoulli(\phi)$$ 

    $$x|y = 0 \approx N(\mu_0, \Sigma_0)$$ 

    $$x|y = 1 \approx N(\mu_1, \Sigma_1)$$ 

    $$p(x; \mu_0, \Sigma_0) = \sqrt{\frac{1}{{(2 \pi)}^n det(\Sigma_0)}} exp(-\frac{1}{2}(x-\mu_0)^T \Sigma_0^{-1}(x-\mu_0))$$

    $$p(x; \mu_1, \Sigma_1) = \sqrt{\frac{1}{{(2 \pi)}^n det(\Sigma_1)}} exp(-\frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1))$$

1. The log-likelihood of the data is given
by 
    $$l(\phi, \mu_0, \mu_1,\Sigma_0, \Sigma_1) = log \prod_{i=1}^m p(x(i), y(i); \phi, \mu_0, \mu_1, \Sigma_0, \Sigma_1)$$

    $$l(\phi, \mu_0, \mu_1,\Sigma_0, \Sigma_1) = log \prod_{i=1}^m p(x(i)| y(i); \phi, \mu_0, \mu_1, \Sigma_0,\Sigma_1)\times p(y(i);\phi)$$ 
    
    By maximizing $l$ with respect to the parameters, we find the maximum likelihood estimate of the parameters (see problem set $1$) to be:

    $$\phi =\frac{1}{m} \sum_{i=1}^m 1\{y(i) = 1\}$$

    $$\mu_0 = \frac{ \sum_{i=1}^m 1\{y(i) = 0\} x(i)}{ \sum_{i=1}^m 1\{y(i) = 0\}}$$

    $$\mu_1 = \frac{\sum_{i=1}^m 1\{y(i) = 1\}x(i)}{\sum_{i=1}^m 1\{y(i) = 1\}}$$

    $$\Sigma_{y^(i)} = \frac{1}{m} \sum_{i=1}^m(x(i) - \mu_{y^(i)})(x(i) - \mu_{y^(i)})^T $$

1. **GDA vs Logistic Regression** : 
GDA works best if $X$ distribution is multivariate. But Logistic Regression works best even for other distributions also. 


# Naive Bayes Classifier

[Source](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html)

## Introduction

1.  In GDA, the feature vectors $X$ were continuous, real-valued vectors. 
1.  Naive Bayes is different learning algorithm in which the $x_j$ ’s are discrete-valued.
1.  This algorithm is mostly used in case of text because words are discrete. 
1.  Naive Bayes works best if we have very less data. 
1.  It tries to model the data and finds out the parameters for the distributions of the classes and based on those parameters, it predicts the new data point.
1.  This is extremely fast because there is no loop or anything. Just need to find out the parameters for the class distributions.
1.  Naive Bayes is a linear classifier. You can find that proof in [Cornell lec 11](https://www.youtube.com/watch?v=GnkDzIOxfzI) in first $10$ mins. 

## Assumptions

All the discrete features are independent of each other given the label.

## Classifier

In case of a binary classifier, $y \in \{0,1\}$ and $X : X_{M\times N}$.  
$$P(X|Y = 0) = P(X_1|Y = 0) \times P(X_2|Y = 0) ........... \times P(X_N|Y = 0) \rightarrow \text{Features are independent}$$
$$P(X|Y = 1) = P(X_1|Y = 1) \times P(X_2|Y = 1) ........... \times P(X_N|Y = 1) $$
Using the Bayes theorem, we can find $P(Y|X)$. 

## Note

Even if the naive bayes assumption violates, this algo works very well. If the naive Bayes assumption holds, then Naive Bayes works similar to Logitstic Regression. [Proof](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html) 

My Comments : This is easy. Watch [Krish Naik Video](https://www.youtube.com/watch?v=jS1CKhALUBQ) for intution and watch [Cornell Video](https://www.youtube.com/watch?v=rqB0XWoMreU) for in-depth understanding.