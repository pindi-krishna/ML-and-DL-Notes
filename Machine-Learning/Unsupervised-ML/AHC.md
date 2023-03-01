# Agglomerative hierarchical clustering (AHC)

## Introduction

1.  The Agglomerative Hierarchical Cluster algorithm is a form of bottom-up clustering, where it starts with an individual element and then groups them into single clusters. 

1.  Hierarchical clustering is often used in the form of descriptive modeling rather than predictive. It doesn’t work well on large datasets, it provides the best results in some cases only.
 
## Algorithm

1.  Each data point is treated as a single cluster. We have $K$ clusters in the beginning. At the start, the number of data points will also be $K$.

1.  Now we need to form a big cluster by joining $2$ closest data points in this step. This will lead to total $K-1$ clusters.

1.  Two closest clusters need to be joined now to form more clusters. This will result in $K-2$ clusters in total. 
1.  Repeat the above three steps until $K$ becomes $0$ to form one big cluster. No more data points are left to join.
1.  After forming one big cluster at last, we can use dendrograms to split the clusters into multiple clusters depending on the use case.

## Advantages

1. AHC is easy to implement, it can also provide object ordering, which can be informative for the display.
1.  We don’t have to pre-specify the number of clusters. It’s easy to decide the number of clusters by cutting the dendrogram at the specific level.
1.  In the AHC approach smaller clusters will be created, which may uncover similarities in data.


## Disadvantages

1.  The objects which are grouped wrongly in any steps in the beginning can’t be undone.
1.  Hierarchical clustering algorithms don’t provide unique partitioning of the dataset, but they give a hierarchy from which clusters can be chosen. 
1.  They don’t handle outliers well. Whenever outliers are found, they will end up as a new cluster, or sometimes result in merging with other clusters.