# Generalized Linear models

[Stanford notes](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf)

## Intro

1.  We’ve seen a regression example, and a classification example. In the regression example, we had $y|x; \theta \approx N(\mu, \sigma^2)$, and in the classification one, $y|x; \theta \approx Bernoulli(\phi)$, for some appropriate definitions of $\mu$ and $\phi$ as functions
of $x$ and $\theta$. In this section, we will show that both of these methods are special cases of a broader family of models, called Generalized Linear Models (GLMs).

1. We say that a class of distributions is in the exponential family if it can be written in the form $$p(y; \eta) = b(y) exp(\eta^T T(y) - a(\eta))$$
    1. $\eta$ - the natural parameter (also called the canonical parameter) of the distribution; 
    2. $T(y)$ - sufficient statistic (for the distributions we consider, it will often be the case that $T(y) = y$)
    3. $a(\eta)$ - log partition function. The quantity $e^{-a(\eta)}$ essentially plays the role of a normalization constant, that makes sure the distribution $p(y; \eta)$ sums/integrates over $y$ to $1$.

1. A fixed choice of $T$, $a$ and $b$ defines a family (or set) of distributions that is parameterized by $\eta$; as we vary $\eta$, we then get different distributions within
this family.

## Distributions in Exponential Family

1.  Bernoulli
1.  Gaussian
1.  Multinoulli

## Bernoulli

1. The Bernoulli distribution with
mean $\phi$, written Bernoulli($\phi$), specifies a distribution over $y \in {0, 1}$, so that $$p(y = 1; \phi) = \phi; p(y = 0; \phi) = 1 - \phi$$

1. As we vary $\phi$, we obtain Bernoulli distributions with different means. We now show that this class of Bernoulli distributions, ones obtained by varying φ, is in the exponential family; i.e., that there is a choice of T, a and b so that above equation becomes exactly the class of Bernoulli distributions. $$p(y; \phi) = \phi^y (1 - \phi)^{1-y} = exp(y log \phi + (1 - y) log(1 - \phi)) $$ $$= exp ((log(\frac{\phi}{1 - \phi}))y + log(1 - \phi))$$
1.  Thus, the natural parameter is given by $\eta = log(\frac{\phi}{(1 - \phi)})$. Interestingly, if
we invert this definition for $\eta$ by solving for $\phi$ in terms of $\eta$, we obtain $ \phi = 1/(1 + e^{-\eta})$. This is the familiar sigmoid function!

1.  The formulation of the Bernoulli distribution as an exponential family distribution,
    $$T(y) = y$$
    $$a(\eta) = -log(1 - \phi) = log(1 + e^{\eta}) - \eta$$
    $$b(y) = 1$$


## Gaussian

1.  When deriving linear regression, the value of $\sigma^2$ had no effect on our final choice of $\theta$ and $h_\theta(x)$.To simplify the derivation below, let’s set $\sigma^2 = 1$. 
$$ p(y;\mu) = \frac{1}{\sqrt(2\pi)} exp(-\frac{1}{2}(y-\mu)^2) = \frac{1}{\sqrt(2\pi)} exp(-\frac{1}{2}y^2 ) exp(\mu y - \frac{1}{2} \mu^2)$$

1.  $\eta = \mu$, 
$T(y) = y$,
$a(\eta) = \frac{\mu^2}{2}$
$b(y) = \frac{1}{\sqrt(2\pi)} exp(−\frac{y^2}{2})$.
1.  If we leave $\sigma^2$ as a variable, the Gaussian distribution can also be shown to be in the
exponential family, where $\eta \in \Re^2$ is now a 2-dimension vector that depends on both $\mu$ and $\sigma$. 

1.  For the purposes of GLMs, however, the $\sigma^2$ parameter can also be treated by considering a more general definition of the exponential family: $$p(y;\mu, \sigma^2 ) = b(a, \tau ) exp((\eta^T T(y) - a(\eta))/c(\tau))$$

1.  Here, $\sigma^2$ is called the dispersion parameter, and for the Gaussian, $c(\tau) = \sigma^2$; but given our simplification above, we won’t need the more general definition for the examples we will consider here.

## Multinoulli
1.  Consider a classification problem in which the response variable y can take on any one of k values, so $y \in {1, 2, . . . , k}$.

1.  To parameterize a multinomial over k possible outcomes, one could use
k parameters $\phi_1, . . . , \phi_k$ specifying the probability of each of the outcomes.

1.  These parameters would be redundant, or more formally, they would not be independent (since knowing any k-1 of the $\phi_i$’s uniquely determines the last one, as they must satisfy $\sum_{i=1}^k \phi_i = 1)$

1.  So, we will instead parameterize the multinomial with only k-1 parameters, $\phi_1, . . . , \phi_{k-1}$

1.  To express the multinomial as an exponential family distribution, we will define $T(y) \in \Re^{k-1}$ as follows:
T(1) = [1 0 0 .....0] T(2) = [0 1 0 0 ... 0] .... T(k) = [0 0 0 ..... 0 ]

1.  We will write $(T(y))_i$ to denote the i-th element of the vector T(y).
$(T(y))_i = 1{y = i}$(if condition is true, it returns 1 otherwise 0).
$$p(y;\phi) = \phi_{1}^{1\{y=1\}} \phi_{2}^{1\{y=2\}}...... \phi_{k}^{1\{y=k\}}$$ 
$$p(y;\phi) = \phi_{1}^{1\{y=1\}} \phi_{2}^{1\{y=2\}}...... \phi_{k}^{1 - \sum_{i=1}^{k} 1\{y=k\}}$$ 

Using $e^{logN} = N$ and $(T(y))_i = 1\{y = i\}$ and simplify :  $$p(y;\phi) = exp((T(y))_{1} log(\frac{\phi_{1}}{\phi_k}) + (T(y))_{2} log(\frac{\phi_{2}}{\phi_k}) +........ + (T(y))_{k-1} log(\frac{\phi_{k−1}}{\phi_k}) + log(\phi_k))$$
$$p(y; \eta) = b(y) exp(\eta^T T(y) - a(\eta))$$
where  $$\eta = [log(\frac{\phi_{1}}{\phi_k}) log(\frac{\phi_{2}}{\phi_k}) ....... log(\frac{\phi_{k-1}}{\phi_k})]^T $$
$$a(\eta) = -log(\phi_k)$$
$$b(y) = 1$$

The link function is given (for i = 1, . . . , k) by
$$\eta_i = log((\frac{\phi_{i}}{\phi_k})) $$

We need to find the response function, $\phi_i$. 
Using $\phi_1 + \phi_2 + .... + \phi_k = 1 $ to give response function : $$\phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^k e^{\eta_j}} $$

This function mapping from the $\eta$’s to the $\phi$’s is called the softmax function.

