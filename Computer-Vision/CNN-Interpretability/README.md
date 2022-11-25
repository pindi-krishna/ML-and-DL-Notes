# CNN Interpretability Techniques

## Prerequisites

### Interpretability vs Explainability

1. Interpretability :  Interpretability is about the extent to which a cause and effect can be observed within a system. Or, to put it another way, it is the extent to which you are able to predict what is going to happen, given a change in input or algorithmic parameters. It’s being able to look at an algorithm and go yep, I can see what’s happening here.

1. Explainability : Explainability, meanwhile, is the extent to which the internal mechanics of a machine or deep learning system can be explained in human terms. 

1. Interpretability is about being able to discern the mechanics without necessarily knowing why. Explainability is being able to quite literally explain what is happening.


### Why Interpretability ?

We must build ‘transparent’ models that explain why they predict what they predict. 

1. First, when the AI is relatively weaker than the human and not yet reliably ‘deployable’, the goal of transparency and explanations is to identify the failure mode.

1. Second, when the AI is on par with humans and reliably ‘deployable’, the goal is to establish appropriate trust and confidence in users.

1. Third, when the AI is significantly stronger than humans, the goal of the explanations is in machine teaching i.e teaching humans how to take better decisions.
 
## Class Activation Maps (CAM)

1. [Youtube Video](https://www.youtube.com/watch?v=vTY58-51XZA)

1. [Github](https://github.com/nickbiso/Keras-Class-Activation-Map)

1. A Class Activation map for a particular category indicates the discriminative region used by CNN to identify the category.

1. The Network mainly consists of a large number of convolutional layers and just before the final output layer, we perform Global Average Pooling. The features thus obtained are fed to a fully connected layer having with softmax activation which produces the desired output. We can identify the importance of the image regions by projecting back the weights of the output layer on the convolutional feature maps obtained from the last Convolution Layer. This technique is known as Class Activation Mapping.

1. The Global Average Pooling layer(GAP) is preferred over the Global MaxPooling Layer(GMP) because GAP layers help to identify the complete extent of the object as compared to GMP layer which identifies just one discriminative part. This is because in GAP we take an average across all the activation which helps to find all the discriminative regions while GMP layer just considers only the most discriminative one.

1. The weights of the final layer corresponding to that class are extracted. Also, the feature map from the last convolutional layer is extracted.

1. Finally, the dot product of the extracted weights from the final layer and the feature map is calculated to produce the class activation map. The class activation map is upsampled by using Bi-Linear Interpolation and superimposed on the input image to show the regions which the CNN model is looking at. 

    ![cam](./Images/CAM.png)

1. Drawbacks : It requires feature maps to directly precede the softmax layers, so it is applicable to a particular kind of CNN architectures that perform global average pooling over convolutional maps immediately before prediction. (i.e conv feature maps $\rightarrow$ global average pooling $\rightarrow$ softmax layer).



## Grad-CAM

1. [Check out demo](http://gradcam.cloudcv.org/)
1. [Watch the demo](https://www.youtube.com/watch?v=COjUB9Izk6E)

### Applications

Grad-CAM is applicable to a wide variety of CNN model-families: 

1. CNNs with fully-connected layers (e.g. VGG)
1. CNNs used for structured outputs (e.g. captioning)
1. CNNs used in tasks with multi-modal inputs (e.g. visual question an- swering) or reinforcement learning, without architectural changes or re-training.


### Introduction

1. Consider image classification – a ‘good’ visual explanation from the model for justifying any target category should be (a) class-discriminative (i.e. localize the category in the image) and (b) high-resolution (i.e. capture fine-grained detail)

1. Past work : Guided Back-propagation and Deconvolution are high-resolution and highlight fine-grained details in the image, but are not class-discriminative.

1. In contrast, localization approaches like CAM or our proposed method Gradient-weighted Class Activation Mapping (Grad-CAM), are highly class-discriminative.

1. Going from deep to shallow layers, the discriminative ability of Grad-CAM significantly reduces as we encounter layers with different output dimensionality and also they lack the ability to show fine-grained importance like pixel-space gradient visualization methods (Guided Back- propagation and Deconvolution).

1. It is possible to fuse existing pixel-space gradient visualizations with Grad-CAM to create Guided Grad-CAM visualizations that are both high-resolution and class discriminative 

1. **Grad-Cam, unlike CAM, uses the gradient information flowing into the last convolutional layer of the CNN** to understand each neuron for a decision of interest because last convolutional layers are expected to have the best compromise between high-level semantics and detailed spatial information.

### Explaination

1. Possibly the most intepretable model — and therefore the one we will use as inspiration — is a regression. In a regression, each feature $x$ is assigned some weight, $w$, which directly tells me that feature’s importance to the model.

1. Specifically, for the $i^{th}$ feature of a specific data point, the feature’s contribution to the model output is $w_i \times x_i$. What does this weight $w$ represent? Well, since a regression is $$Y = w_1 \times x_1 + w_2 \times x_2 + ........+ w_n \times x_n + b$$ Then, $$w_i = \frac{\partial Y}{\partial x_i}$$

1. In other words, the weight assigned to the $i^{th}$ feature tells us the gradient of that feature with respect to the model’s prediction: how the model’s prediction changes as the feature changes.
Conveniently, this gradient is easy to calculate for neural networks. So, in the same way that for a regression, a feature’s contribution is
    $$w_i \times x_i = x_i \times \frac{\partial Y}{\partial x_i}$$
, perhaps the gradient can be used to explain the output of a neural network.

## Problem with Gradients

[Medium Blog](https://towardsdatascience.com/interpretable-neural-networks-45ac8aa91411)

There are two issues we run into when trying to use the above approach:
1. Firstly, feature importances are relative. For gradient boosted decision trees, a feature’s shap value tells me how a feature changed the prediction of the model relative to the model not seeing that feature. Since neural networks can’t handle null input features, we’ll need to redefine a feature’s impact relative to something else.

1. To overcome this, we’ll define a new baseline: what am I comparing my input against? One example is for the MNIST digits dataset. Since all the digits are white, against a black background, perhaps a reasonable background would be a fully black image, since this represents no information about the digits. Choosing a background for other datasets is much less trivial — for instance, what should the background be for the ImageNet dataset? We’ll discuss a solution for this later, but for now let’s assume that we can find a baseline for each dataset.

1. The second issue is that using the gradient of the output with respect to the input works well for a linear model — such a regression — but quickly falls apart for nonlinear models. To see why, let’s consider a “neural network” consisting only of a ReLU activation, with a baseline input of $x=2$.
Now, lets consider a second data point, at $x = -2$. $ReLU(x=2) = 2$, and $ReLU(x=-2) = 0$, so my input feature $x = -2$ has changed the output of my model by $2$ compared to the baseline. This change in the output of my model has to be attributed to the change in $x$, since its the only input feature to this model, but the gradient of $ReLU(x)$ at the point $x = -2$ is $0$! This tells me the contribution of $x$ to the output is $0$, which is obviously a contradiction.

1. This happened for two reasons: firstly, **we care about a finite difference in the function (the difference between the function when $x = 2$ and when $x = -2$), but gradients calculate infinitesimal differences**. Secondly, the ReLU function can get saturated — once $x$ is smaller than $0$, it doesn’t matter how much smaller it gets, since the function will only output $0$. As we saw above, this results in inconsistent model intepretations, where the output changes with respect to the baseline, but no features are labelled as having caused this change. 

1. These inconsistencies are what Integrated Gradients and DeepLIFT attempt to tackle. They both do this by recognizing that ultimately, what we care about is not the gradient at the point $x$; we care about **how the output changed from the baseline as the input changed from the baseline.**

## Fundamental axioms which needs to be satisfied to be a attribution method

**Axiom**

1. Sensitivity(a) : An attribution method satisfies Sensitivity(a) if for every input and baseline that differ in one feature but have different predictions then the differing feature should be given a non-zero attribution. 

1. Implementation Invariance : Two networks are functionally equivalent if their outputs are equal for all inputs, despite having very different implementations. Attribution methods should satisfy Implementation Invariance, i.e., the attributions are always identical for two functionally equivalent networks. 

If an attribution method fails to satisfy the above axioms, the attributions are potentially sensitive to unimportant aspects of the models.

### Gradients

1. Gradients (of the output with respect to the input) is a natural analog of the model coefficients for a deep network, and therefore the product of the gradient and feature values is a reasonable starting point for an attribution method.

1. Gradients violate Sensitivity(a): For a concrete example, consider a one variable, one ReLU network, $f(x) = 1 - ReLU(1-x)$. Suppose the baseline is $x = 0$ and the input is $x = 2$. The function changes from $0 to 1$, but because $f$ becomes flat at $x = 1$, the gradient method gives attribution of $0 to x$. Intuitively, gradients break Sensitivity because the prediction function may flatten at the input and thus have zero gradient despite the function value at the input being different from that at the baseline.
 
## Integrated Gradients

[Blog](https://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html)

### Introduction

1. The approach taken by integrated gradients is to ask the following question: what is something I can calculate which is an analogy to the gradient which also acknowledges the presence of a baseline?

1. Part of the problem is that the gradient (of the output relative to the input) at the baseline is going to be different than the gradient at the output I measure; ideally, I would consider the gradient at both points. This is not enough though: consider a sigmoid function, where my baseline output is close to $0$, and my target output is close to $1$:

    ![selfattention](./Images/sigmoid_ig.png)

1. In the above figure, both the gradient at  baseline and at data point are going to be close to $0$. In fact, all the interesting — and informative — gradients are in between the two data points, so ideally we would find a way to capture all of that information too.

1. That’s exactly what Integrated Gradients do, by calculating the integral of the gradients between the baseline and the point of interest. Actually calculating the integral of the gradients is intractable, so instead, they are approximated using a Reimann sum: the gradients are taken at lots of small steps between the baseline and the point of interest.

### Algorithm

1. Start from the baseline where baseline can be a black image whose pixel values are all zero or an all-white image, or a random image. Baseline input is one where the prediction is neutral and is central to any explanation method and visualizing pixel feature importances.

1. Generate a linear interpolation between the baseline and the original image. Interpolated images are small steps$(\alpha)$ in the feature space between your baseline and input image and consistently increases with each interpolated image’s intensity. 
    $$\gamma(\alpha) = x' + \alpha \times (x - x')$$  
    
    for $\alpha \in [0, 1]$ where $\gamma(\alpha)$ is the interpolated image $\&$ and $x'$ -- Baseline image and $x$ -- Input image

1. Calculate gradients to measure the relationship between changes to a feature and changes in the model’s predictions.

1. Compute the numerical approximation through averaging gradients.

1. Scale IG to the input image to ensure that the attribution values are accumulated across multiple interpolated images are all in the same units. Represent the IG on the input image with the pixel importances.

[Tensorflow implementation](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)

### How to pick a baseline

1. As promised, we will now return to picking baseline. Aside from some very obvious cases (eg. the MNIST example above), deciding what the baseline inputs are is extremely non trivial, and might require domain expertise.

1. An alternative to manually picking a baseline is to consider what the prior distribution of a trained model is. This can give us a good idea of what the model is thinking when it has no information at all.

1. For instance, if I have trained a model on the ImageNet, what is its prior assumption going to be that a new photo it sees if of a meerkat? If $2\%$ of the photos in the ImageNet dataset are of a meerkat, then the model is going to think that the new photo it sees has a $2\%$ chance of being a meerkat. It will then adjust its prediction accordingly when it actually sees the photo. It makes sense to measure the impact of the inputs relative to this prior assumption.

1. So how can I pick a baseline which is $2\%$ meerkat? Well, a good approach might be to take the mean of the dataset, simply by averaging the images in the dataset together. This is the approach used in the shap library’s implementations of Integrated Gradients and DeepLIFT. Conveniently, it removes the need to be a domain expert to pick an appropriate baseline for the model being interpreted.