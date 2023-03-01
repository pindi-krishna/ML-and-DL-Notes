# Grad-CAM

1. [Check out demo](http://gradcam.cloudcv.org/)
1. [Watch the demo](https://www.youtube.com/watch?v=COjUB9Izk6E)

## Applications

Grad-CAM is applicable to a wide variety of CNN model-families: 

1. CNNs with fully-connected layers (e.g. VGG)
1. CNNs used for structured outputs (e.g. captioning)
1. CNNs used in tasks with multi-modal inputs (e.g. visual question an- swering) or reinforcement learning, without architectural changes or re-training.


## Introduction

1. Consider image classification – a ‘good’ visual explanation from the model for justifying any target category should be (a) class-discriminative (i.e. localize the category in the image) and (b) high-resolution (i.e. capture fine-grained detail)

1. Past work : Guided Back-propagation and Deconvolution are high-resolution and highlight fine-grained details in the image, but are not class-discriminative.

1. In contrast, localization approaches like CAM or our proposed method Gradient-weighted Class Activation Mapping (Grad-CAM), are highly class-discriminative.

1. Going from deep to shallow layers, the discriminative ability of Grad-CAM significantly reduces as we encounter layers with different output dimensionality and also they lack the ability to show fine-grained importance like pixel-space gradient visualization methods (Guided Back- propagation and Deconvolution).

1. It is possible to fuse existing pixel-space gradient visualizations with Grad-CAM to create Guided Grad-CAM visualizations that are both high-resolution and class discriminative 

1. **Grad-Cam, unlike CAM, uses the gradient information flowing into the last convolutional layer of the CNN** to understand each neuron for a decision of interest because last convolutional layers are expected to have the best compromise between high-level semantics and detailed spatial information.

## Explaination

1. Possibly the most intepretable model — and therefore the one we will use as inspiration — is a regression. In a regression, each feature $x$ is assigned some weight, $w$, which directly tells me that feature’s importance to the model.

1. Specifically, for the $i^{th}$ feature of a specific data point, the feature’s contribution to the model output is $w_i \times x_i$. What does this weight $w$ represent? Well, since a regression is $$Y = w_1 \times x_1 + w_2 \times x_2 + ........+ w_n \times x_n + b$$ Then, $$w_i = \frac{\partial Y}{\partial x_i}$$

1. In other words, the weight assigned to the $i^{th}$ feature tells us the gradient of that feature with respect to the model’s prediction: how the model’s prediction changes as the feature changes.
Conveniently, this gradient is easy to calculate for neural networks. So, in the same way that for a regression, a feature’s contribution is
    $$w_i \times x_i = x_i \times \frac{\partial Y}{\partial x_i}$$
, perhaps the gradient can be used to explain the output of a neural network.

# Problem with Gradients

[Medium Blog](https://towardsdatascience.com/interpretable-neural-networks-45ac8aa91411)

There are two issues we run into when trying to use the above approach:
1. Firstly, feature importances are relative. For gradient boosted decision trees, a feature’s shap value tells me how a feature changed the prediction of the model relative to the model not seeing that feature. Since neural networks can’t handle null input features, we’ll need to redefine a feature’s impact relative to something else.

1. To overcome this, we’ll define a new baseline: what am I comparing my input against? One example is for the MNIST digits dataset. Since all the digits are white, against a black background, perhaps a reasonable background would be a fully black image, since this represents no information about the digits. Choosing a background for other datasets is much less trivial — for instance, what should the background be for the ImageNet dataset? We’ll discuss a solution for this later, but for now let’s assume that we can find a baseline for each dataset.

1. The second issue is that using the gradient of the output with respect to the input works well for a linear model — such a regression — but quickly falls apart for nonlinear models. To see why, let’s consider a “neural network” consisting only of a ReLU activation, with a baseline input of $x=2$.
Now, lets consider a second data point, at $x = -2$. $ReLU(x=2) = 2$, and $ReLU(x=-2) = 0$, so my input feature $x = -2$ has changed the output of my model by $2$ compared to the baseline. This change in the output of my model has to be attributed to the change in $x$, since its the only input feature to this model, but the gradient of $ReLU(x)$ at the point $x = -2$ is $0$! This tells me the contribution of $x$ to the output is $0$, which is obviously a contradiction.

1. This happened for two reasons: firstly, **we care about a finite difference in the function (the difference between the function when $x = 2$ and when $x = -2$), but gradients calculate infinitesimal differences**. Secondly, the ReLU function can get saturated — once $x$ is smaller than $0$, it doesn’t matter how much smaller it gets, since the function will only output $0$. As we saw above, this results in inconsistent model intepretations, where the output changes with respect to the baseline, but no features are labelled as having caused this change. 

1. These inconsistencies are what Integrated Gradients and DeepLIFT attempt to tackle. They both do this by recognizing that ultimately, what we care about is not the gradient at the point $x$; we care about **how the output changed from the baseline as the input changed from the baseline.**

# Fundamental axioms which needs to be satisfied to be a attribution method

**Axiom**

1. Sensitivity(a) : An attribution method satisfies Sensitivity(a) if for every input and baseline that differ in one feature but have different predictions then the differing feature should be given a non-zero attribution. 

1. Implementation Invariance : Two networks are functionally equivalent if their outputs are equal for all inputs, despite having very different implementations. Attribution methods should satisfy Implementation Invariance, i.e., the attributions are always identical for two functionally equivalent networks. 

If an attribution method fails to satisfy the above axioms, the attributions are potentially sensitive to unimportant aspects of the models.

## Gradients

1. Gradients (of the output with respect to the input) is a natural analog of the model coefficients for a deep network, and therefore the product of the gradient and feature values is a reasonable starting point for an attribution method.

1. Gradients violate Sensitivity(a): For a concrete example, consider a one variable, one ReLU network, $f(x) = 1 - ReLU(1-x)$. Suppose the baseline is $x = 0$ and the input is $x = 2$. The function changes from $0 to 1$, but because $f$ becomes flat at $x = 1$, the gradient method gives attribution of $0 to x$. Intuitively, gradients break Sensitivity because the prediction function may flatten at the input and thus have zero gradient despite the function value at the input being different from that at the baseline.