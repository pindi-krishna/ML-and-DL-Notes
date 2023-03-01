# Decision Trees

## Motivation
In case of KD trees, If you knew that a test point falls into a cluster of 1 million points with all positive label, you would know that its neighbors will be positive even before you compute the distances to each one of these million distances. It is therefore sufficient to simply know that the test point is an area where all neighbors are positive, its exact identity is irrelevant.

## Introduction

1. Decision trees are exploiting exactly that. Here, we do not store the training data, instead we use the training data to build a tree structure that recursively divides the space into regions with similar labels. 
1. The root node of the tree represents the entire data set. This set is then split roughly in half along one dimension by a simple threshold $t$. All points that have a feature value $≥t$ fall into the right child node, all the others into the left child node. The threshold $t$ and the dimension are chosen so that the resulting child nodes are purer in terms of class membership. 
1. Ideally all positive points fall into one child node and all negative points in the other. If this is the case, the tree is done. If not, the leaf nodes are again split until eventually all leaves are pure (i.e. all its data points contain the same label) or cannot be split any further (in the rare case with two identical points of different labels).  

## Parametric or Non-Parametric

1. Decision Trees are also an interesting case. If they are trained to full depth they are non-parametric, as the depth of a decision tree scales as a function of the training data (in practice $O(log2n)$). 
1. If we however limit the tree depth by a maximum value they become parametric (as an upper bound of the model size is now known prior to observing the training data). We can also split on the same feature multiple times. 

## Advantages

1.  Once the tree is constructed, the training data does not need to be stored. Instead, we can simply store how many points of each label ended up in each leaf - typically these are pure so we just have to store the label of all points; 
1.  Decision trees are very fast during test time, as test inputs simply need to traverse down the tree to a leaf - the prediction is the majority label of the leaf
1.  Decision trees require no metric because the splits are based on feature thresholds and not distances.
1.  Best for Interpretability. 

## Limitations

1.  Splitting the tree until each and every point in the training set is correct because leads to overfitting. 
1.  *Decision boundaries are always parallel to the axes.*
1.  To find out the best split, impurity function for all the possible splits in all the possible features have to be tried and consider the split with lowest impurity. I guess, it is very slow in case of continuous features. 

## Impurity Functions

Data: $S=\{(x_1,y_1),…,(x_n,y_n)\},y_i \in \{1,…,c\}$, where $c$ is the number of classes. 
1. Gini impurity: Let $S_k \subset S$ where $S_k=\{(x,y) \in S:y=k\}$ (all inputs with labels $k$) $S=S_1\cup S_2 \cup ..... S_c$

Define: $$p_k = \frac{|S_k|}{|S|} $$ -- Fraction of inputs in $S$ with label $k$.
Gini Impurity : 
$$G(S) = \sum_{k=1}^c p_k(1-p_k)$$
Gini impurity of a tree:
$$GT(S)=\frac{|S_L|}{|S|}GT(S_L)+\frac{|S_R|}{|S|}GT(S_R)$$
where:
$$S=S_L\cup S_R$$
$$S_L\cap S_R=\phi $$

$\frac{|S_L|}{|S|}$ -- fraction of inputs in left substree $\&$
$\frac{|S_R|}{|S|}$ -- fraction of inputs in right substree

Entropy : Let $p_1,…,p_k$ be defined as before. We know what we don't want (Uniform Distribution): $p_1=p_2=.....=p_c=\frac{1}{c}$ This is the worst case since each leaf is equally likely. Prediction is random guessing. Define the impurity as how close we are to uniform. 

$$H(S) = \sum_{k=1}^C p_klog(p_k)$$ 
$$H(S)=p_LH(S_L)+p_RH(S_R)$$
where $p_L=\frac{|S_L|}{|S|}$,$p_R=\frac{|S_R|}{|S|}$

## ID3 Algorithm

ID3 algorithm stop under two cases. 
1. The first case is that all the data points in a subset have the same label. If this happens, we should stop splitting the subset and create a leaf with label $y$. 
1. The other case is there are no more attributes could be used to split the subset. Then we create a leaf and label it with the most common $mode(y)$ in case of classification and $mean(y)$ in case of regression.

## How to split

1. Try all features and all possible splits. Pick the split that minimizes impurity (e.g. $s>t$) where $f$←feature and $t$←threshold
1. In case of continuous features, if shape of data : $(N \times d)$. Split between every two points in all the dimensions. Therefore, $N-1$ splits, $d$ dimensions gives us $d\times(N-1)$ splits. 

## Drawback

Decision trees has a bias and variance problem and finding a sweet spot between them is hard. That is why in real life, they don't work very well. 
To solve the variance problem, we use bagging and can be used in any algorithm. 

## Incase of Regression

Incase of regression, we compute squared loss instead of Gini index and Entropy. After each split we will compute variance on the left and right trees and split when the weighted variance is **low**. Stopping criteria would be either the limit set for the depth or split until every leaf contains one datapoint. 

Average squared difference from average label $$L(S)=\frac{1}{|S|}\sum_{(x,y) \in S} (y-\overline{y})^2$$ 
where $\overline{y_S} =\frac{1}{|S|} \sum_{(x,y)\in S} y$ 
That is why decision trees are also called as CART (Classification and Regression trees).
