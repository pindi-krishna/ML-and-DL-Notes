# Computer Vision
1. [Stanford Notes](https://cs231n.github.io/convolutional-networks/)
# Introduction

## Why CNN?

Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size $32 \times 32 \times 3$ ($32$ wide, $32$ high, $3$ color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have $32 \times 32 \times 3 = 3072$ weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. $200 \times 200 \times 3$, would lead to neurons that have $200 \times 200 \times 3 = 120,000$ weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

## Convolution Neural Network

As we described above, a simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.

#### Sample CNN Architecture

1. **INPUT** $[32 \times 32 \times 3]$ will hold the raw pixel values of the image, in this case an image of width $32$, height $32$, and with $3$ color channels R,G,B.

1. **CONV** layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as $[32 \times 32 \times 12]$ if we decided to use $12$ filters.

1. **RELU** layer will apply an elementwise activation function, such as the $max(0,x)$ thresholding at zero. This leaves the size of the volume unchanged $([32 \times 32 \times 12])$.

1. **POOL** layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as $[16 \times 16 \times 12]$.

1. **FC** (i.e. fully-connected) layer will compute the class scores, resulting in volume of size $[1 \times 1 \times 10]$, where each of the $10$ numbers correspond to a class score, such as among the $10$ categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.


Each Layer may or may not have parameters (e.g. $CONV/FC$ do, $RELU/POOL$ don’t)

Each Layer may or may not have additional hyperparameters (e.g. $CONV/FC/POOL$ do, $RELU$ doesn’t).

## Convolutional Layer

### Introduction

1. The CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. 
1. For example, a typical filter on a first layer of a ConvNet might have size $5 \times 5 \times 3$ (i.e. $5$ pixels width and height, and $3$ because images have depth $3$, the color channels). During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a $2$-dimensional activation map that gives the responses of that filter at every spatial position. 
1. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. $12$ filters), and each of them will produce a separate $2$-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.
    $$Output Volume = \frac{(W-F+2P)}{S}+1$$
where, input volume size $(W)$, the receptive field size or filter size $(F)$, the stride $(S)$, and padding $(P)$. 
To ensure, output size is same as input size, we use Zero-Padding (i,e) $P = (F -1)/2$ and $stride = 1$. 

**Note**  
* Zero-Padding doesn't mean $P = 0$. It means $P = (F -1)/2$. 
* $P = 0$ is called as Valid-Padding.

### Constraints on strides

Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size $W=10$, no zero-padding is used $P=0$, and the filter size is $F=3$, then it would be impossible to use stride $S=2$, since $(W-F+2P)/S+1=(10-3+0)/2+1=4.5$, i.e. not an integer, indicating that the neurons don’t “fit” neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library could throw an exception or zero pad the rest to make it fit, or crop the input to make it fit, or something. As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions “work out” can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.

### Parameter Sharing
1. Real world example: The Krizhevsky et al. architecture that won the ImageNet challenge in 2012 accepted images of size $[227 \times 227 \times 3]$. On the first Convolutional Layer, it used neurons with receptive field size $F=11$, stride $S=4$ and no zero padding $P=0$ (This is called as "Valid" padding). Since $(227 - 11)/4 + 1 = 55$, and since the Conv layer had a depth of $K=96$, the Conv layer output volume had size $[55 \times 55 \times 96]$. Each of the $55 \times 55 \times 96$ neurons in this volume was connected to a region of size $[11 \times 11 \times 3]$ in the input volume. Moreover, all $96$ neurons in each depth column are connected to the same $[11 \times 11 \times 3]$ region of the input, but of course with different weights.

1. Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are $55 \times 55 \times 96 = 290,400$ neurons in the first Conv Layer, and each has $11 \times 11 \times 3 = 363$ weights and $1$ bias. 

1. Together, this adds up to $290400  \times  364 = 105,705,600$ parameters on the first layer of the ConvNet alone. Clearly, this number is very high.

1. It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position $(x_1,y_1)$, then it should also be useful to compute at a different position $(x_2,y_2)$. 

1. In other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size $[55 \times 55 \times 96]$ has $96$ depth slices, each of size $[55 \times 55]$), we are going to constrain the neurons in each depth slice to use the same weights and bias. 

1. With this parameter sharing scheme, the first Conv Layer in our example would now have only $96$ unique set of weights (one for each depth slice), for a total of $96 \times 11 \times 11 \times 3 = 34,848$ unique weights, or $34,944$ parameters ($+96$ biases). 

1. Alternatively, all $55 \times 55$ neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.

1. If detecting a horizontal edge is important at some location in the image, it should intuitively be useful at some other location as well due to the translationally-invariant structure of images. 

1. There is therefore no need to relearn to detect a horizontal edge at every one of the $55*55$ distinct locations in the Conv layer output volume.


### Summary of Convolutional Layer

1. Accepts a volume of size $W1 \times H1 \times D1$ 

1. Requires four hyperparameters:
    1. Number of filters $K$, 
    1. their spatial extent $F$,
    1. the stride $S$,
    1. the amount of padding $P$.

1. Produces a volume of size $W2 \times H2 \times D2$ where: 

    1. $W2=\frac{(W1-F+2P)}{S}+1$ 
    1. $H2=\frac{(H1-F+2P)}{S}+1$ (i.e. width and height are computed equally by symmetry)
    1. $D2=K$

1. With parameter sharing, it introduces $F⋅F⋅D1$  weights per filter, for a total of $(F⋅F⋅D1)⋅K$ weights and $K$ biases.

1. In the output volume, the $d$-th depth slice (of size $W2 \times H2$) is the result of performing a valid convolution of the $d$-th filter over the input volume with a stride of $S$, and then offset by $d$-th bias.

1. A common setting of the hyperparameters is $F=3,S=1,P=1$.

### Backpropagation

The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters).

## Pooling Layer

### Introduction

1. It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. 
1. **The Pooling Layer operates independently on every depth slice of the input** and resizes it spatially, using the MAX operation. 
1. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding $75\%$ of the activations. The depth dimension remains unchanged. 

### Summary

1. Accepts a volume of size $W1 \times H1 \times D1$.

1. Requires two hyperparameters:
    1. Spatial extent F
    1. Stride S

1. Produces a volume of size $W2 \times H2 \times D2$ where: $W2=(W1-F)/S+1$ and  $H2=(H1-F)/S+1$ and $D2=D1$. 
1. Introduces zero parameters since it computes a fixed function of the input.
1. For Pooling layers, it is not common to pad the input using zero-padding


### Getting rid of pooling

Many people dislike the pooling operation and think that we can get away without it. For example, Striving for Simplicity: The All Convolutional Net proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.

# CNN Architectures

### Introduction

#### Layer Patterns

1. In other words, the most common ConvNet architecture follows the pattern:
    $$INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC$$
    Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). 

1. $$INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC$$ 
    Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation. 


#### Why we prefer a stack of small filter CONV to one large receptive field CONV layer?

1. Suppose that you stack three $3 \times 3$ CONV layers on top of each other (with non-linearities in between, of course). 

1. In this arrangement, each neuron on the first CONV layer has a $3 \times 3$ view of the input volume. A neuron on the second CONV layer has a $3 \times 3$ view of the first CONV layer, and hence by extension a $5 \times 5$ view of the input volume. Similarly, a neuron on the third CONV layer has a $3 \times 3$ view of the 2nd CONV layer, and hence a $7 \times 7$ view of the input volume. 

1. Suppose that instead of these three layers of $3 \times 3$ CONV, we only wanted to use a single CONV layer with $7 \times 7$ receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent $(7 \times 7)$, but with several disadvantages. 

1. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. 

1. Second, if we suppose that all the volumes have $C$ channels, then it can be seen that the single $7 \times 7$ CONV layer would contain $C \times (7 \times 7 \times C) = 49C^2$ parameters, while the three $3 \times 3$ CONV layers would only contain $3 \times (C \times (3 \times 3 \times C)) = 27C^2$ parameters. 

1. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. 

1. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.


#### Practical Suggestion
Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data (Don't try to be a hero -- Andrej Karpathy). 

### Designing a CNN Architecture

#### Input Layer
Must be divisible by $2$ many times. Common numbers include $32$ (e.g. CIFAR-10), $64$, $96$ (e.g. STL-10), or $224$ (e.g. common ImageNet ConvNets), $384$, and $512$.

#### CONV Layers

1. The conv layers should be using small filters (e.g. $3 \times 3$ or at most $5 \times 5$), using a stride of $S=1$, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. 

1. For a general $F$, it can be seen that $P=(F−1)/2$ preserves the input size. If you must use bigger filter sizes (such as $7 \times 7$ or so), it is only common to see this on the very first conv layer that is looking at the input image.


#### POOL Layers

1. The pool layers are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with $2 \times 2$ receptive fields (i.e. $F=2$), and with a stride of $2$ (i.e. $S=2$).

1. Note that this discards exactly $75\%$ of the activations in an input volume (For every $2 \times 2 = 4$ pixels, we are considering $1$ max-value which other $3$ non-maximum values are being discarded). 

1. Another slightly less common setting is to use $3 \times 3$ receptive fields with a stride of $2$, but this makes “fitting” more complicated (e.g., a $32 \times 32 \times 3$ layer would require zero padding to be used with a max-pooling layer with $3 \times 3$ receptive field and stride $2$). 

1. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.


#### Reducing sizing headaches

The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don’t zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters “work out”, and that the ConvNet architecture is nicely and symmetrically wired.

#### Why use stride of $1$ in CONV?

Smaller strides work better in practice. Additionally, as already mentioned stride $1$ allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

#### Why use padding?

In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly.

#### Compromising based on memory constraints

In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a $224 \times 224 \times 3$ image with three $3 \times 3$ CONV layers with $64$ filters each and padding $1$ would create three activation volumes of size $[224 \times 224 \times 64]$. This amounts to a total of about 10 million activations, or $72$ MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of $7 \times 7$ and stride of $2$ (as seen in a ZF net). As another example, an AlexNet uses filter sizes of $11 \times 11$ and stride of $4$.

#### Practical Computational Considerations

There are three major sources of memory to keep track of:


1. From the intermediate volume sizes: These are the raw number of activations at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.

1. From the parameter sizes: These are the numbers that hold the network parameters, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so (parameters, its gradients, it momentum history etc).
Every ConvNet implementation has to maintain miscellaneous memory, such as the image data batches, perhaps their augmented versions, etc.


### Dense CNN

#### Why?

1. The advantage of ResNets is that the gradient can flow di-rectly through the identity function from later layers to the earlier layers. However, the identity function and the output of any layer are combined by summation, which may impede the information flow in the network.
1. Therefore, To further improve the information flow  between  layers this architecture has been developed. 
1. This ensures maximum information and gradient flow between layers in the network. 

#### DenseNET

1. **Dense connectivity** : Figure 1 illustrates the layout ofthe  resulting  DenseNet  schematically. Consequently,  the'th layer receives the feature-maps of all preceding layers, $x0,...,x'−1$ as input: $x'=H'([x0,x1,...,x'−1])$, where $[x0,x1,...,x'−1]$ refers to the concatenation of the feature-maps produced in layers 0,...,−1. Because of itsdense connectivity we refer to this network architecture asDense Convolutional Network (DenseNet). 

1. **Composite function** : We define $H'(l)$ as  a  composite  function  of  three  consecutive  operations : batch normalization (BN), followed by a ReLU and a 3×3 Conv layers. 

1. **Pooling** : The concatenation operation used   in the above Eq. is not viable when the size of feature-maps changes. However,  an  essential  part  of  convolutional  networks  is down-sampling layers that change the size of feature-maps.T o facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks. Layers between blocks are transition layers,  which do convolution and pooling. The transition layers $$[BN -> 1 \times 1 conv -> 2×2 Avg pooling layer]$$

1. **Growth  rate(K)** (hyper parameter) : If  each  function $H`(l)$ produces k feature-maps, it follows that the $l^{th}$layer has $k_0 + k \times (l−1)$ input feature-maps, where $k_0$ is the number of channels in the input layer.  An important difference between DenseNet and existing  network  architectures is that DenseNet can have very narrow layers,e.g.,k= 12.

1. **Bottleneck layers[DenseNet-B]** : Although each layer only produces k output feature-maps, it typically has many more inputs.  It has been noted that a 1×1 convolution can be introduced as bottle neck layer before each 3×3 convolution to  reduce  the  number  of  input  feature-maps,  and  thus  to improve computational efficiency.  We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer,i.e. [BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)]. In our experiments, we let each 1×1 convolution produce 4k feature-maps. 

1. **Compression[DenseNet-C]** : To  further  improve  model  compactness, we  can  reduce  the  number  of  feature-maps  at  transition layers.   If a dense block contains $m$ feature-maps,  we let the following transition layer generate $\theta \times m$ output feature-maps, where $0< \theta \leq 1$ is referred to as the compression factor.  When $θ=1$ , the number of feature-maps across transition layers remains unchanged.

#### Feature  Reuse 
Observations : 

1. All layers spread their weights over many inputs within the same block.  This indicates that features extracted by very early layers are, indeed, directly used by deep layers throughout the same dense block.
1. The weights of the transition layers also spread their weight  across  all  layers  within  the  preceding  dense block, indicating information flow from the first to the last layers of the DenseNet through few in directions.
1. The  layers  within  the  second  and  third  dense  block consistently assign the least weight to the outputs of the transition layer (the top row of the triangles), indicating that the transition layer outputs many redundant features (with low weight on average).  This is in keeping with the strong results of DenseNet-BC[Both Bottle Neck and Compression] where exactly these outputs are compressed.
1. Although  the  final  classification  layer,  shown  on  the very  right,  also  uses  weights  across  the  entire  dense block, there seems to be a concentration towards final feature-maps, suggesting that there may be some more high-level features produced late in the network


#### Conclusion

1. DenseNet introduces direct connections between any two layers with the same feature-map size.

1. DenseNets require  substantially  fewer  parameters  and  less  computation to achieve state-of-the-art performances.

1. DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved.

1. [Github link](https://github.com/facebook/fb.resnet.torch)








