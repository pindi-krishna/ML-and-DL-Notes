# Bagging
1. [Cornell Notes](https://www.cs.cornell.edu/courses/cs4780/2021fa/lectures/lecturenote18.html)
1. [MIT Slides](https://people.csail.mit.edu/dsontag/courses/ml13/slides/lecture13.pdf)

## Why

To reduce Variance. This can be used with all the algorithms suffering with variance problem. 

## Introduction

$$Variance = E[h_D(x) - \overline{h(x)}]^2$$
where $\overline{h(x)}$ is the expected classifier that is average of all the classifiers trained on different datasets drawn from same distribution and $h_D(x)$ is the classifier trained on a single dataset. 

1. To reduce the variance, we need to reduce difference between and  $h_D(x)$ and $\overline{h(x)}$.

1. The weak law of large numbers says (roughly) for $I.I.D$ random variables $x_i$ with mean $\overline{x}$, we have, 
$\frac{1}{m}\sum_{i=1}^m x_i\rightarrow \overline{x} $ as $m\rightarrow \infty$. 

1. Apply this to classifiers: Assume we have $m$ training sets $D_1,D_2,...,D_m$ drawn from probability distribution $P$. Train a classifier on each one and average result: 
    $$\widehat{h}=\frac{1}{m}\sum_{i=1}^{m}h_{D_i}\rightarrow \overline{h} \quad \text{as}  \quad m \rightarrow \infty$$

1. We refer to such an average of multiple classifiers as an ensemble of classifiers.

1. Good news: If $\widehat{h}\rightarrow \overline{h}$ the variance component of the error must also vanish.

1. Problem:
We don't have $m$ data sets $D_1,....,D_m$, we only have $D$.


## Algorithm
![bagging](./Images/Bagging.png)
1. Sample $m$ data sets $D_1,....,D_m$ from $D$ **with replacement**.
1. For each $D_j$ train a classifier $h_j(x)$
1. Classify new instance by majority vote / average.

In practice larger $m$ results in a better ensemble, however at some point you will obtain diminishing returns. 
*Note that setting $m$ unnecessarily high will only slow down your classifier but will not increase the error of your classifier.*

**Math**

Each data point has probability ${(1 – 1/n)}^n$
of being selected as test data. [Proof](https://juanitorduz.github.io/bootstrap/)
1. What is the probability that the first bootstrap observation is not the $j^{th}$ observation from the original sample?
    * As the probability of selecting a particular $x_j$ from the set $x_1, x_2,...,x_n$, is $1/n$ then the desired probability is $(1 – 1/n)$. 
1. What is the probability that the second bootstrap observation is not the $j^{th}$ observation from the original sample?
    * It would be $(1 – 1/n) \times (1 – 1/n)$ because the selections are independent.
1. If we are sampling $n$ times, then probability of $j^th$ observation not being part of the dataset would ${(1 – 1/n)}^n$. 

Training data = $1- {(1 – 1/n)}^n$ of the original data
## Advantages

1.  Easy to implement
1.  Reduces variance. 
1.  As the prediction is an average of many classifiers, we obtain a mean score and variance. Latter can be interpreted as the uncertainty of the prediction.
1.  No need to split the train set into training and validation further. The idea is that each training point was not picked in all the data sets $D_k$. If we average the classifiers $h_k$ of all such data sets, we obtain a classifier (with a slightly smaller $m$) that was not trained on $(x_i,y_i)$ ever and it is therefore equivalent to a test sample. If we compute the error of all these classifiers **which doesn't contain that particular point**, we obtain an estimate of the true test error. The beauty is that we can do this without reducing the training set. We just run bagging as it is intended and obtain this so called out-of-bag error for free.

## How do we justify having duplicates in the dataset when bagging, given that duplicates can cause bias towards duplicated observations over single observations in most models?

[Quora Link](https://www.quora.com/How-do-we-justify-having-duplicates-in-the-dataset-when-bagging-given-that-duplicates-can-cause-bias-towards-duplicated-observations-over-single-observations-in-most-models)

Having duplicates is actually desirable. Note that bagging is an ensemble technique — you sample multiple datasets from the original dataset, and fit a model to each of these new datasets.

When these new training sets contain duplicates, the corresponding model puts more weight on getting these examples correct. So, different models will focus on different training points, and when you take a combination of all of them, you’re likely to get a better result than that from a model which is trying to focus on all the data at once.

If you have sufficiently many new datasets, each training point will be duplicated with equal probability, so you are not changing the distribution of data.


# Random Forest

A Random Forest is essentially nothing else but bagged decision trees, with a slightly modified splitting criteria.

## Algorithm

1. The algorithm works as follows:
Sample $m$ data sets $D_1,…,D_m$ from $D$ **with replacement**.
1. For each $D_j$ train a full decision tree $h_j$ $(depth_{max}=\infty)$   with one small modification: before each split randomly subsample $k \leq d$ features (**without replacement**) and only consider these for your split. (This further increases the variance of the trees.)
1. The final classifier is $$h(x)=\frac{1}{m}\sum_{j=1}^{m}h_j(x)$$

## Advantages

1. The RF only has two hyper-parameters, $m$ and $k$. It is extremely insensitive to both of these. A good choice for $k$ is $k=\sqrt{d}$ (where $d$ denotes the number of features). 

1. Decision trees do not require a lot of preprocessing. For example, the features can be of different scale, magnitude, or slope. This can be highly advantageous in scenarios with heterogeneous data, for example the medical settings where features could be things like blood pressure, age, gender, ..., each of which is recorded in completely different units.

## Why would we sample features without replacement and data points with replacement ?