# KNN Algorithm

## Assumption

Similar Inputs have similar outputs which imply that data points of various classes are not randomly sprinkled across the space, but instead appear in clusters of more or less homogeneous class assignments.

## Classification rule

For a test input $x$, assign the most common label amongst its $k$ most similar training inputs.

## What distance function should we use to find neighbors?

The k-nearest neighbor classifier fundamentally relies on a distance metric. The better that metric reflects label similarity, the better the classified will be. 

1. Minkowski distance between two points $x$ and $x'$: $$D(x, x') = {(\sum_{i=1}^d {(x'_i - x_i)^p})}^\frac{1}{p}$$
where $d$ represents the number of dimensions. The most common choice is the 
    1.  $p = 1$  (Manhattan distance)
    1.  $p = 2$  (Euclidean distance)
    1.  $p = \infty$ (Max of difference between coordinates in each dimension)

1. Cosine Similarity :
$$Cosine Similarity(x, x') = \frac{x.x'}{|x||x'|}$$
Generally, incase of classification of documents(bag of words), then it is better to consider cosine similarity.

**Note : We need to choose distance metric wisely based on our problem**

## How to choose k

1.  Generally we go for odd $K$. 
1.  In case of multi class classification with four classes $C1, C2, C3, C4$ with distribution of (say $K = 7$) nearest neighbors $(3,3,1,0)$, If $C1$ and $C2$ have same number of neighbors, then we can check all its majority classes and assign that class which contains the closest point.  

## What if $K = n$

Then the output of all test points will be same as major class in the given data set irrespective of its input. Therefore, never ever choose $K = n$.

## What if $K = 1$

1.  The $1-NN$ could be an outlier of some another class. Then it miss-classifies the point. Therefore, its better not to choose. 
1.  As $n\rightarrow \infty$, the $1-NN$($1$ Nearest Neighbor) classifier is only a factor $2$ worse than the best possible classifier. 

1. Let $x_{NN}$ be the nearest neighbor of our test point $x_t$. As $n\rightarrow \infty, dist(x_{NN},x_{t}) \rightarrow 0, (i.e.) x_{NN} \rightarrow x_t$. (This means the nearest neighbor is identical to $x_t$.) You return the label of $x_{NN}$. What is the probability that this is not the label of $x_t$? (This is the probability of drawing two different label of $x$). Solving this probability, we reach to the above conclusion. 

## Curse of Dimensionality

1.  As the dimension $d$ of the input increases, the distance between two points in the space also increases. 
1.  So as $d >> 0$ almost the entire space is needed to find the $10$-NN. 
1.  This breaks down the KNN assumptions, because the $KNN are not particularly closer (and therefore more similar) than any other data points in the training set. Why would the test point share the label with those k-nearest neighbors, if they are not actually similar to it?
1.  Dont try to visualize the above point as it involves multiple dimensions greater than three. Do the math.
1.  Note : In real life, points are not uniformly distributed. Points mostly lie on the complex manifolds and may form clusters. That's why KNN may work sometimes even for higher dimensions. 

## Pros and Cons

Pros :
1.  Easy and simple 
1.  Non parametric 
1.  No assumption about data. 
1.  Only two choices ($k$ values and distance metric).

Cons : 
Computation time : $O(N.d)$ where $N$ -> #training samples and $d$ -> #dimensions.

## K Dimensional Tress (KD Trees)

1.  Building KD trees. 

    1. Split data recursively in half on exactly one feature.
    1. Rotate through features.
        1. When rotating through features, a good heuristic is to pick the feature with maximum variance.

    Max height of the tree could be $log_2(n)$.

1. Finding NN for the test point(X,y). 

    1. Find region containing (x,y). 
    1. Compare to all points in the region.

1.  How can this partitioning speed up testing?

    1.  Let's think about it for the one neighbor case.
    1.  Identify which side the test point lies in, e.g. the right side.
    1.  Find the nearest neighbor ${x_{NN}}^R$ of $x_t$ in the same side. The $R$ denotes that our nearest neighbor is also on the right side.
    1.  Compute the distance between $x_y$ and the dividing "wall". Denote this as $d_w$. If $d_w>d(x_t,{x_{NN}}^R)$ you are done, and we get a $2\times$ speedup.

1.  Pros: Exact and  Easy to build.
1.  Cons:
    1.  Curse of Dimensionality makes KD-Trees ineffective for higher number of dimensions. May not work better if dimensions are greater than 10. 
    1.  All splits are axis aligned (all dividing hyperplanes are parallel to axis).

1.  Approximation: Limit search to $m$ leafs only. 

## Ball Trees

1.  Similar to KD-trees, but instead of boxes use hyper-spheres (balls). If the distance to the ball, $d_b$, is larger than distance to the currently closest neighbor, we can safely ignore the ball and all points within.

1. The ball structure allows us to partition the data along an underlying manifold that our points are on, instead of repeatedly dissecting the entire feature space (as in KD-Trees).

1. Ball trees allows us to split along the dimension with maximum variance instead of splitting along the feature axis in half. 

1.  Construction :

    ![BTC](./Images/ball_tree_construction.png)

1.  Ball-Tree Use : 
    1.  Same as KD-Trees
    1.  Slower than KD-Trees in low dimensions $(d \leq 3)$ but a lot faster in high dimensions. Both are affected by the curse of dimensionality, but Ball-trees tend to still work if data exhibits local structure (e.g. lies on a low-dimensional manifold).

## Locally sensitive hashing (LSH) 
    
1. Divide the whole space of points into $\frac{n}{2^k}$ regions by randomly drawing k hyperplanes $(h_1, h_2, h_3, .........,h_k)$. 
1. Compare $x$ to only those $\frac{n}{2^k}$ points in that particular region. 
1. Complexity : $O(Kd +  d \frac{n}{2^k})$. 
    1. $Kd$ : To find out which point belongs to which region. For that we need to check with each hyperplane and find that. 
    1. $d\frac{n}{2^k}$ : Finding the NN by comparing d-dimensional point with $\frac{n}{2^k}$ points. 
1. Limitations : Choosing the right set and number of hyperplanes are really important. Try a couple of different initializations. Based on the way we choose the hyperplanes, we may miss out the main neighbors and misclassify the point. 
    
## Conclusion

1.  KNN is best suitable if dimensions are low and high number of data points. 
1.  KNN works better incase of images and faces though the data is highly dimensoinal due to its sparsity. Moreover, similar points similar labels assumption kind of stays true.
1.  More the number of data points, slow is the algorithm. 
