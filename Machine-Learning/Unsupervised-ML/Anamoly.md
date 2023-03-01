# Anomaly detection Techniques

## Isolation Forest (BASIC INTRO)

1.  Isolation forest is a machine learning algorithm for anomaly detection.
1.  It's an unsupervised learning algorithm that identifies anomaly by isolating outliers in the data.
1.  Isolation Forest is based on the Decision Tree algorithm. It isolates the outliers by randomly selecting a feature from the given set of features and then randomly selecting a split value between the max and min values of that feature. This random partitioning of features will produce shorter paths in trees for the anomalous data points, thus distinguishing them from the rest of the data.
1.  Isolation Forest isolates anomalies in the data points instead of profiling normal data points. As anomalies data points mostly have a lot shorter tree paths than the normal data points, trees in the isolation forest does not need to have a large depth so a smaller $max\_depth$ can be used resulting in low memory requirement.
1.  Using Isolation Forest, we can detect anomalies faster and also require less memory compared to other algorithms.
1.  We can use sklearn.ensemble.IsolationForest 