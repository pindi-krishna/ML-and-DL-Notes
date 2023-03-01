# Gaussian Mixture Model (GMM)

## Introduction

1.  GMM can be used to find clusters in the same way as K-Means. The probability that a point belongs to the distribution’s center decreases as the distance from the distribution center increases. The bands show a decrease in probability in the below image. 

1.  Since GMM contains a probabilistic model under the hood, we can also find the probabilistic cluster assignment. 
1.  *When you don’t know the type of distribution in data, you should use a different algorithm*.

## Parameters

A GMM is a function composed of several Gaussians, each identified by $K$, where $K$ is the number of clusters. Each Gaussian $K$ in the mixture is comprised of following parameters:

1.  A mean $\mu$ that defines its center.
1.  A covariance $\sum$ that defines its width.


## Algorithm
The mean and variance for each gaussian is determined using a technique called Expectation-Maximization (EM).

## Expectation-Maximization in Gaussian Mixture Models

1.  We typically use EM when the data has missing values, or in other words, when the data is incomplete.
1.  Expectation :  
For each point $x_i$, calculate the probability that it belongs to cluster/distribution $c1, c2, … ck$. This is done using the below formula:
    $$r_{ic} = \frac{P(x_i \in c_i)}{\sum_j P(x_i \in c_j) ....}$$
    This value will be high when the point is assigned to the correct cluster and lower otherwise.
1.  The mean and the covariance matrix are updated based on the values assigned to the distribution, in proportion with the probability values for the data point. Hence, a data point that has a higher probability of being a part of that distribution will contribute a larger portion:
$$\mu_c = \frac{1}{N_c}\sum_i r_{ic}x_i$$
$$\sum_c = \frac{1}{N_c} \sum_i r_{ic}(x_i - \mu_c)^T(x_i - \mu_c) $$

## Advantages

1.  One of the advantages of GMM over K-Means is that K-Means doesn’t account for variance (here, variance refers to the width of the bell-shaped curve) and GMM returns the probability that data points belong to each of K clusters. 
1.  In case of overlapped clusters, all the above clustering algorithms fail to identify it as one cluster. 
1.  GMM uses a probabilistic approach and provides probability for each data point that belongs to the clusters. 


## Disadvantages

1.  Mixture models are computationally expensive if the number of distributions is large or the dataset contains less observed data points.
1.  It needs large datasets and it’s hard to estimate the number of clusters.

