# Supervised Machine Learning Algorithms

The goal is, given a training set, to learn a function $h : X \rightarrow Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function $h$ is called a hypothesis.

# Terminology

## Parametric vs Non-parametric algorithms

1. A parametric algorithm is one that has a constant set of parameters, which is **independent** of the number of training samples. You can think of it as the amount of much space you need to store the trained classifier. An examples for a parametric algorithm is the Perceptron algorithm, or logistic regression. Their parameters consist of $w,b$, which define the separating hyperplane. The dimension of $w$ depends on the number of dimensions of the training data, but not on how many training samples you use for training.

1. In contrast, the number of parameters of a non-parametric algorithm scales as a function of the training samples. An example of a non-parametric algorithm is the $K$-Nearest Neighbors classifier. Here, during "training" we store the entire training data -- so the parameters that we learn are identical to the training set and the number of parameters (the storage we require) grows linearly with the training set size.

