# Integrated Gradients

[Blog](https://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html)

## Introduction

1. The approach taken by integrated gradients is to ask the following question: what is something I can calculate which is an analogy to the gradient which also acknowledges the presence of a baseline?

1. Part of the problem is that the gradient (of the output relative to the input) at the baseline is going to be different than the gradient at the output I measure; ideally, I would consider the gradient at both points. This is not enough though: consider a sigmoid function, where my baseline output is close to $0$, and my target output is close to $1$:

    ![selfattention](./Images/sigmoid_ig.png)

1. In the above figure, both the gradient at  baseline and at data point are going to be close to $0$. In fact, all the interesting — and informative — gradients are in between the two data points, so ideally we would find a way to capture all of that information too.

1. That’s exactly what Integrated Gradients do, by calculating the integral of the gradients between the baseline and the point of interest. Actually calculating the integral of the gradients is intractable, so instead, they are approximated using a Reimann sum: the gradients are taken at lots of small steps between the baseline and the point of interest.

## Algorithm

1. Start from the baseline where baseline can be a black image whose pixel values are all zero or an all-white image, or a random image. Baseline input is one where the prediction is neutral and is central to any explanation method and visualizing pixel feature importances.

1. Generate a linear interpolation between the baseline and the original image. Interpolated images are small steps$(\alpha)$ in the feature space between your baseline and input image and consistently increases with each interpolated image’s intensity. 
    $$\gamma(\alpha) = x' + \alpha \times (x - x')$$  
    
    for $\alpha \in [0, 1]$ where $\gamma(\alpha)$ is the interpolated image $\&$ and $x'$ -- Baseline image and $x$ -- Input image

1. Calculate gradients to measure the relationship between changes to a feature and changes in the model’s predictions.

1. Compute the numerical approximation through averaging gradients.

1. Scale IG to the input image to ensure that the attribution values are accumulated across multiple interpolated images are all in the same units. Represent the IG on the input image with the pixel importances.

[Tensorflow implementation](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)

## How to pick a baseline

1. As promised, we will now return to picking baseline. Aside from some very obvious cases (eg. the MNIST example above), deciding what the baseline inputs are is extremely non trivial, and might require domain expertise.

1. An alternative to manually picking a baseline is to consider what the prior distribution of a trained model is. This can give us a good idea of what the model is thinking when it has no information at all.

1. For instance, if I have trained a model on the ImageNet, what is its prior assumption going to be that a new photo it sees if of a meerkat? If $2\%$ of the photos in the ImageNet dataset are of a meerkat, then the model is going to think that the new photo it sees has a $2\%$ chance of being a meerkat. It will then adjust its prediction accordingly when it actually sees the photo. It makes sense to measure the impact of the inputs relative to this prior assumption.

1. So how can I pick a baseline which is $2\%$ meerkat? Well, a good approach might be to take the mean of the dataset, simply by averaging the images in the dataset together. This is the approach used in the shap library’s implementations of Integrated Gradients and DeepLIFT. Conveniently, it removes the need to be a domain expert to pick an appropriate baseline for the model being interpreted.