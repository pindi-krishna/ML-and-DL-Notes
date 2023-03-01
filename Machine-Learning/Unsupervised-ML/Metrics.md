# Metrics

## Silhouette score

1.  Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of $[-1, 1]$.
1.  A value of $+1$ indicates sample is far way from the neighboring clusters. A value of $0$ indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
1.  The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
1.  Formula
    $$sihouette score(s) = \frac{b - a}{
    max(a,b)}$$
    where $a$: The mean distance between a sample and all other points in the same class. \
    $b$: The mean distance between a sample and all other points in the next nearest cluster.
1.  Limitations :
The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.

## Davies Bouldin Index (DBI)

The index is defined as the average similarity between each cluster 
$C_i$ for $i = 1,2,.. ,k$ and its most similar one $C_j$. In the context of this index, similarity is defined as a measure $R_{ij}$ that trades off:

1.  $s_i$, the average distance between each point of cluster $i$ and the centroid of that cluster – also know as cluster diameter.
1.  $d_{ij}$, the distance between cluster centroids i and j.

A simple choice to construct $R_{ij}$ so that it is nonnegative and symmetric is:
$$R_{ij} = \frac{s_i + s_j}{d_{ij}}$$

Then the Davies-Bouldin index is defined as:
$$DB = \sum_{i = 1}^k max_{i\neq j} R_i$$