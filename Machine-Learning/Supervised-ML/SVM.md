# Support Vector Machines(SVM)

1.  Consider a positive training example ($y = 1$). The larger $\theta^T x$ is, the larger also is $h(x) = p(y = 1|x;w, b)$, and thus also the higher our degree of “confidence” that the label is 1.

1.  Again informally it seems that we’d have found a good fit to
the training data if we can find $\theta$ so that $\theta^T x(i) >> 0$ whenever y(i) = 1, and $\theta^T x(i) << 0 $ whenever $y(i) = 0$, since this would reflect a very confident (and correct) set of classifications for all the training examples.

1. We’ll use $y \in \{-1, 1\}$ (instead of $\{0, 1\}$) to denote the class labels. $$h_{w,b}(x) = g(w^T x + b)$$
Here, $g(z) = 1$ if $z \geq 0$, and $g(z) = -1$ otherwise.
$w$ takes the role of $[\theta_1, . . . ,\theta_n]^T$ and $b$ is 
intercept term.

1. Note also that, from our definition of $g$ above, our classifier will directly predict either $1$ or $-1$, without first going through the intermediate step of estimating the probability of $y$ being $1$.

1. Given a training example ($x(i), y(i)$), we define the functional margin of ($w, b$) with respect to the training example $$\gamma(i) = y(i)(w^T x(i) + b)$$ 

1. If $y(i)(w^T x(i) + b) > 0$, then our prediction on this example is correct.

1. Given our choice of $g$, we note that if we replace $w$ with $2w$ and b with $2b$, then since $g(w^Tx+b) = g(2w^Tx+2b)$,this would not change $h_{w,b}(x)$ at all. (i.e.), $g$, and hence also $h_{w,b}(x)$, depends only on the sign, but not on the magnitude of $w^T x + b$. However, replacing $(w, b)$ with $(2w, 2b)$ also results in multiplying our functional margin by a factor of $2$.

1. Intuitively, it might therefore make sense to impose some sort of normalization condition such as that $||w||_2 = 1$ i.e., we might replace $(w, b)$ with $(w/||w||_2, b/||w||_2)$.

1. Given a training set $S = \{(x(i), y(i)); i = 1, . . . ,m\}$, we also define the functional margin of $(w, b)$ with respect to $S$ as the smallest of the functional margins of the individual training examples $$\hat{\gamma}(i) = min_{i=1,...,m} \gamma(i)$$

1. Geometric Margins : The geometric margin of $(w, b)$ with respect to a training example $(x(i), y(i))$ to be $$\hat{\gamma}(i) = y(i)(\frac{w^T}{||w||} x(i) + \frac{b}{||w||})$$

1. Note that if $||w|| = 1$, then the functional margin equals the geometric margin — this thus gives us a way of relating these two different notions of margin.

1.  Specifically, because of this invariance to the scaling of the parameters, when trying to fit $w$ and $b$ to training data, we can impose an arbitrary scaling constraint on $w$ without changing anything important; for instance, we can demand that $||w|| = 1$, or $|w1| = 5$, or $|w1 + b| + |w2| = 2$, and any of these can be satisfied simply by rescaling $w$ and $b$.

## Objective 

$$max_{w,b} \gamma$$ 
s.t  $y(i)(w^T x(i) + b) \geq \hat{\gamma}$, for  $i = 1, . . . ,m$ and $||w|| = 1$ 

1. (i.e.), we want to maximize $\gamma$, subject to each training example having functional margin at least $\hat{\gamma}$. The $||w|| = 1$ constraint moreover ensures that the
functional margin equals to the geometric margin, so we are also guaranteed that all the geometric margins are at least $\hat{\gamma}$.

1. If we could solve the optimization problem above, we’d be done. But the “$||w|| = 1$” constraint is a nasty (non-convex) one, and this problem certainly
isn’t in any format that we can plug into standard optimization software to
solve. So, let’s try transforming the problem into a nicer one $$max_{w,b} \frac{\gamma}{||w||}$$
s.t. $y(i)(w^T x(i) + b) \geq \hat{\gamma}$ for  $i = 1, . . . ,m$.

1. Here, we’re going to maximize $\frac{\hat{\gamma}}{||w|}$, subject to the functional margins all being at least $\hat{\gamma}$. Recall our earlier discussion that we can add an arbitrary scaling constraint on $w$ and $b$ without changing anything. This is the key idea we’ll use now.

1. We will introduce the scaling constraint that the
functional margin of $w$, $b$ with respect to the training set must be $1$: $$\hat{\gamma} = 1$$

1. Since multiplying $w$ and $b$ by some constant results in the functional margin being multiplied by that same constant, this is indeed a scaling constraint, and can be satisfied by rescaling $w, b$. Plugging this into our problem above,
and noting that maximizing $$\frac{\gamma^\wedge}{||w||} = \frac{1}{||w||}$$ is the same thing as minimizing $||w||^2$, we now have the following optimization problem: $$min_{w,b} ||w||^2$$
s.t. $y(i)(w^T x(i) + b) \geq 1$ for  $i = 1, . . . ,m$ 
1. The above is an optimization problem with a convex quadratic objective and only linear constraints. Its solution gives us the optimal margin classifier.

*Note : The mathematical derivation of SVM can be found [here](https://iiitaphyd-my.sharepoint.com/:b:/g/personal/krishna_chandra_research_iiit_ac_in/EWGaLL7gMy9Ht68sJW--RmgBP8n5wbJ2TALp2bFNLRtQ_g?e=G7vngX)
and the clarity will be given in [cornell lec14](https://www.youtube.com/watch?v=xpHQ6UhMlx4&list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS&index=14) at $35:00$.* 

## What if the data is not linearly separable?

1. In this case, the constraint is never satisfied and we can never find the hyperplane. Therefore, a slack variable $\epsilon$ has been introduced. Now the objective and constraint is slightly modified so that the constraint is satisfied, 

    $$min_{w,b} ||w||^2 + C.\sum_i \epsilon_i$$
    s.t. $y(i)(w^T x(i) + b) \geq 1$ and $\epsilon_i \geq 0$ for all $i$ and $\epsilon_i = max(0, 1 - y(i)(w^T x(i) + b))$.

1. $C$ is a hyperparameter set by us. This penalizes the outliers or the datapoints which doesn't satisfy the constraints. For $C$, try from $10^{-4}$ to $100$ and check which one works better. For clear understanding, look at figure below. 

    ![SVMC](./Images/SVM_C.png)

## Parametric vs Non-parametric algorithms

An interesting edge case is kernel-SVM. Here it depends very much on kernel we are using. e.g. linear SVMs are parametric (for the same reason as the Perceptron or logistic regression). So if the kernel is linear the algorithm is clearly parametric. However, if we use an RBF kernel then we cannot represent the classifier of a hyper-plane of finite dimensions. Instead we have to store the support vectors and their corresponding dual variables $\alpha_i$ -- the number of which is a function of the data set size (and complexity). Hence, the kernel-SVM with an RBF kernel is non-parametric. A strange in-between case is the polynomial kernel. It represents a hyper-plane in an extremely high but still finite-dimensional space. So technically one could represent any solution of an SVM with a polynomial kernel as a hyperplane in an extremely high dimensional space with a fixed number of parameters, and the algorithm is therefore (technically) parametric. However, in practice this is not practical. Instead, it is almost always more economical to store the support vectors and their corresponding dual variables (just like with the RBF kernel). It therefore is technically parametric but for all means and purposes behaves like a non-parametric algorithm.

## Important Q $\&$ A
1. Why shouldn't we incorporate bias as a constant features when working with SVMs ?
    1. If we include bias in the datapoint, then when maximizing the margin, we are trying to minimize the $||w||^2 + b^2$ which changes the objective function and this is not what we want.  In the formula of distance from point to plane, we dont include bias in the denominator. therefore, bias will not play any role in the objective function, but it plays in the satisfying the constraint. 