# Computer Vision

# Introduction

### Why CNN?
Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32*32*3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. 200x200x3, would lead to neurons that have 200*200*3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.
### Convolution Neural Network}
As we described above, a simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.
#### Sample CNN Architecture}

1. \textbf{INPUT} [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
1. \textbf{CONV} layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
1. \textbf{RELU} layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
1. \textbf{POOL} layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
1. \textbf{FC} (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.


Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)
Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t).

### Convolutional Layer}
#### Introduction}
The CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5 pixels width and height, and 3 because images have depth 3, the color channels). During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.
$$\text{Output Volume} = \frac{(W-F+2P)}{S}+1$$
where, input volume size (W), the receptive field size or filter size (F), the stride (S), and padding (P). 
To ensure, output size is same as input size, we use Zero-Padding (i,e) P = (F -1)/2 and stride = 1. 
#### Constraints on strides}
Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size $W=10$, no zero-padding is used P=0, and the filter size is $F=3$, then it would be impossible to use stride $S=2$, since $(W-F+2P)/S+1=(10-3+0)/2+1=4.5$, i.e. not an integer, indicating that the neurons don’t “fit” neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library could throw an exception or zero pad the rest to make it fit, or crop the input to make it fit, or something. As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions “work out” can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.
#### Parameter Sharing}
\begin{itemize}
1. Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are $55*55*96 = 290,400$ neurons in the first Conv Layer, and each has $11*11*3 = 363$ weights and 1 bias. 
1. Together, this adds up to $290400 * 364 = 105,705,600$ parameters on the first layer of the ConvNet alone. Clearly, this number is very high.

1. It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). 
1. In other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. 
1. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96*11*11*3 = 34,848 unique weights, or 34,944 parameters (+96 biases). 
1. Alternatively, all $55*55$ neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.
1. If detecting a horizontal edge is important at some location in the image, it should intuitively be useful at some other location as well due to the translationally-invariant structure of images. 
1. There is therefore no need to relearn to detect a horizontal edge at every one of the $55*55$ distinct locations in the Conv layer output volume.
\end{itemize}
#### Summary of Convolutional Layer}

1. Accepts a volume of size $W1 \times H1 \times D1$ 
1. Requires four hyperparameters:
1. Number of filters K, their spatial extent F,
the stride S,the amount of zero padding P.
1. Produces a volume of size $W2 \times H2 \times D2$ where: 

    1. $W2=\frac{(W1-F+2P)}{S}+1$ 
    1. $H2=\frac{(H1-F+2P)}{S}+1$ (i.e. width and height are computed equally by symmetry)
    1. $D2=K$

1. With parameter sharing, it introduces F⋅F⋅D1  weights per filter, for a total of (F⋅F⋅D1)⋅K
weights and K biases.
1. In the output volume, the d-th depth slice (of size $W2 \times H2$) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.
1. A common setting of the hyperparameters is F=3,S=1,P=1.


#### Backpropagation}
The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters).
### Pooling Layer}
#### Introduction}

1. It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. 
1. \textbf{The Pooling Layer operates independently on every depth slice of the input} and resizes it spatially, using the MAX operation. 
1. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding $75\%$ of the activations. The depth dimension remains unchanged. 

#### Summary}

1. Accepts a volume of size W1×H1×D1
1. Requires two hyperparameters:
their spatial extent F, the stride S,
1. Produces a volume of size W2×H2×D2
where: $W2=(W1-F)/S+1$ and 
$H2=(H1-F)/S+1$ and $D2=D1$. 
1. Introduces zero parameters since it computes a fixed function of the input.
1. For Pooling layers, it is not common to pad the input using zero-padding


#### Getting rid of pooling}
Many people dislike the pooling operation and think that we can get away without it. For example, Striving for Simplicity: The All Convolutional Net proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.
# CNN Architectures}
### Introduction}
#### Layer Patterns}

1. In other words, the most common ConvNet architecture follows the pattern:
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). 

1. INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC Here we see two CONV layers stacked before every POOL layer. This is generally a \textbf{good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation}


#### Why we prefer a stack of small filter CONV to one large receptive field CONV layer}

\begin{itemize}
1. Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). 
1. In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. 
1. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. 
1. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. 
1. Second, if we suppose that all the volumes have C channels, then it can be seen that the single 7x7 CONV layer would contain $C×(7×7×C)=49C^2$ parameters, while the three 3x3 CONV layers would only contain $3×(C×(3×3×C))=27C^2$ parameters. 
1. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. 
1. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.
\end{itemize}

#### Practical Suggestion}
Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data (Don't try to be a hero -- Andrej Karpathy). 

### Designing a CNN Architecture}

#### Input Layer}
Must be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.

#### CONV Layers}
\begin{itemize}
1. The conv layers should be using small filters (e.g. 3x3 or at most 5x5), using a stride of S=1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. 
1. For a general F, it can be seen that P=(F−1)/2 preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.
\end{itemize}

#### POOL Layers}
\begin{itemize}
1. The pool layers are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2).

1. Note that this discards exactly $75\%$ of the activations in an input volume (For every 2x2 = 4 pixels, we are considering 1 max-value which other 3 non-maximum values are being discarded). 
1. Another slightly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes “fitting” more complicated (e.g., a 32x32x3 layer would require zero padding to be used with a max-pooling layer with 3x3 receptive field and stride 2). 
1. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.
\end{itemize}

#### Reducing sizing headaches} 
The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don’t zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters “work out”, and that the ConvNet architecture is nicely and symmetrically wired.

#### Why use stride of 1 in CONV?} 
Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

#### Why use padding?} 
In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly.

#### Compromising based on memory constraints}
In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filter sizes of 11x11 and stride of 4.

#### Practical Computational Considerations}
There are three major sources of memory to keep track of:


1. From the intermediate volume sizes: These are the raw number of activations at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.

1. From the parameter sizes: These are the numbers that hold the network parameters, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so (parameters, its gradients, it momentum history etc).
Every ConvNet implementation has to maintain miscellaneous memory, such as the image data batches, perhaps their augmented versions, etc.


### Dense CNN}
#### Why}

    1. The advantage of ResNets is that the gradient can flow di-rectly through the identity function from later layers to the earlier layers. However, the identity function and the output of any layer are combined by summation, which may impede the information flow in the network.
    1. Therefore, To further improve the information flow  between  layers this architecture has been developed. 
    1. This ensures maximum information and gradient flow between layers in the network. 

#### DenseNET}

    1. \textbf{Dense connectivity} : Figure 1 illustrates the layout ofthe  resulting  DenseNet  schematically.   Consequently,  the`thlayer receives the feature-maps of all preceding layers,x0,...,x`−1, as input:x`=H`([x0,x1,...,x`−1]),(2)where[x0,x1,...,x`−1]refers to the concatenation of thefeature-maps produced in layers0,...,`−1. Because of itsdense connectivity we refer to this network architecture asDense Convolutional Network (DenseNet). 
    1. \textbf{Composite function} : We define $H'(l)$ as  a  composite  function  of  three  consecutive  operations : batch normalization (BN), followed by a ReLU and a 3×3 Conv layers. 
    1. \textbf{Pooling} : The concatenation operation used   in the above Eq. is not viable when the size of feature-maps changes. However,  an  essential  part  of  convolutional  networks  is down-sampling layers that change the size of feature-maps.T o facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks. Layers between blocks are transition layers,  which do convolution and pooling. The transition layers [BN -> $1\times 1$ conv -> 2×2 Avg pooling layer].
    1. \textbf{Growth  rate(K)} (hyper parameter) : If  each  function $H`(l)$ produces k feature-maps, it follows that the $l^{th}$layer has $k_0 + k \times (l−1)$ input feature-maps, where $k_0$ is the number of channels in the input layer.  An important difference between DenseNet and existing  network  architectures is that DenseNet can have very narrow layers,e.g.,k= 12.
    1. \textbf{Bottleneck layers[DenseNet-B]} : Although each layer only produces k output feature-maps, it typically has many more inputs.  It has been noted that a 1×1 convolution can be introduced as bottle neck layer before each 3×3 convolution to  reduce  the  number  of  input  feature-maps,  and  thus  to improve computational efficiency.  We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer,i.e. [BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)]. In our experiments, we let each 1×1 convolution produce 4k feature-maps. 
    1. \textbf{Compression[DenseNet-C]} : To  further  improve  model  compactness, we  can  reduce  the  number  of  feature-maps  at  transition layers.   If a dense block contains $m$ feature-maps,  we let the following transition layer generate $\theta \times m$ output feature-maps, where $0< \theta \leq 1$ is referred to as the compression factor.  When $θ=1$ , the number of feature-maps across transition layers remains unchanged.

#### Feature  Reuse} \newline
Observations : 

    1. All layers spread their weights over many inputs within the same block.  This indicates that features extracted by very early layers are, indeed, directly used by deep layers throughout the same dense block.
    1. The weights of the transition layers also spread their weight  across  all  layers  within  the  preceding  dense block, indicating information flow from the first to the last layers of the DenseNet through few in directions.
    1. The  layers  within  the  second  and  third  dense  block consistently assign the least weight to the outputs of the transition layer (the top row of the triangles), indicating that the transition layer outputs many redundant features (with low weight on average).  This is in keeping with the strong results of DenseNet-BC[Both Bottle Neck and Compression] where exactly these outputs are compressed.
    1. Although  the  final  classification  layer,  shown  on  the very  right,  also  uses  weights  across  the  entire  dense block, there seems to be a concentration towards final feature-maps, suggesting that there may be some more high-level features produced late in the network


#### Conclusion}

    1. DenseNet introduces direct connections between any two layers with the same feature-map size.
    1. DenseNets require  substantially  fewer  parameters  and  less  computation to achieve state-of-the-art performances.
    1. DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved.
    1. Github link : https://github.com/facebook/fb.resnet.torch




# GANs}
#### Introduction}

1. The main focus for GAN (Generative Adversarial Networks) is to generate data from scratch, mostly images but other domains including music have been done.

1. GAN composes of two deep networks, the generator, and the discriminator. Generator generates the images similar to training data under the guidance of Discriminator. 

1. First, we sample some noise z using a normal or uniform distribution. With z as an input, we use a generator G to create an image $(x=G(z))$. 

#### Non Convergence}
GAN is a game where your opponent always counteracts your actions. The optimal solution is known as Nash equilibrium which is hard to find. Gradient descent is not necessarily a stable method for finding such equilibrium. When mode collapses, the training turns into a cat-and-mouse game in which the model will never converge. Just another thought, maybe the nature of the game makes GANs hard to converge.

# Capsule Networks}
Main source : https://pechyonkin.me/capsules-2/
### Why}
#### Limitations of Convolutional and Pooling operations}

1. CNNs have a few drawbacks in recognizing features of input data when they are in different orientations.  CNNs are not perfect.
1.  Max pooling that consecutively selects the largest number in each region(kernel). As result, we get what we wanted — invariance of activities. 
1. \textbf{Invariance} means that by changing the input a little, the output still stays the same.
1. When in the input image we shift the object that we want to detect by a little bit, networks activities (outputs of neurons) will not change because of max pooling and the network will still detect the object.
1. The above described mechanism is not very good, because max pooling loses valuable information and also does not encode relative spatial relationships between features.
1. Geoffrey Hinton stated in one of his lectures that,
"The pooling operation used in convolutional neural networks is a big mistake, and the fact that it works so well is a disaster!".

### Capsules}
#### Capsule}

1. A Capsule Network is a neural network that tries to perform Inverse graphics.
1. Capsules encapsulate all important information about the state of the feature they are detecting in vector form.
1. One key feature of capsule network is that they preserve the detailed info about the orientation and its pose.
1. Capsules encode probability of detection of a feature as the length of their output vector. And the state of the detected feature is encoded as the direction in which that vector points to (“instantiation parameters”). So when detected feature moves around the image or its state somehow changes, the probability still stays the same (length of vector does not change), but its orientation changes.


#### How Capsules Work}

1. 


### Summary}

1. A Capsule Network could be considered as a ‘real-imitation’ of the human brain. Unlike Convolutional Neural Networks that do not evaluate the spatial relationships in the given data, Capsule Networks consider the orientation of images to be significant in analyzing the data.
1. They examine the hierarchical relationships to better identify images. The Inverse-graphics mechanism which our brains make use of, to build a hierarchical representation of an image to match it with what we’ve learned, is what drives a Capsule Network to show remarkable results. 
1. Though it isn’t yet computationally efficient, the accuracy does seem beneficial in tackling real-world scenarios. The Dynamic Routing of Capsules is what makes all of this possible.
1. It employs an unusual strategy of updating the weights in a network, thus avoiding the pooling operation. As time passes by, Capsule Networks shall penetrate into various other fields, making machines more human-like.

# Object Detection and Segmentation}
### Introduction}

1. Object detection is the task of finding the different objects in an image and classifying them.
1. \textbf{Region-CNN} :  The goal of R-CNN is to take in an image, and correctly identify where the main objects (via a bounding box) in the image.
\textbf{Inputs:} Image \newline
\textbf{Outputs:} Bounding boxes [4 corners co-ordinates] + labels for each object in the image.
1. Detection pipeline : Region Proposal generator $\rightarrow$ Feature extractor $\rightarrow$ Classification. 


### Prerequisites}
#### Selective Search Method}
This Algorithm looks at the image through windows of different sizes, and for each size tries to group together adjacent pixels by texture, color, or intensity to identify objects.

\begin{figure*}
    \centering
        \includegraphics [width=1\textwidth, height = 0.25\textheight]{Diagrams/RCNN.png}
    \caption{After creating a set of region proposals, R-CNN passes the image through a modified version of AlexNet to determine whether or not it is a valid region}
    \label{fig:rcnn}
\end{figure*}

### R-CNN} \newline
 R-CNN is just the following steps \ref{fig:rcnn}:

1. The first module generates 2,000 region proposals using the Selective Search algorithm.
1. After being resized to a fixed pre-defined size, the second module extracts a feature vector of length 4,096 from each region proposal.
1. The third module uses a pre-trained SVM algorithm to classify the region proposal to either the background or one of the object classes.


#### Drawbacks}

1. It is a multi-stage model, where each stage is an independent component. Thus, it cannot be trained end-to-end.
1. It caches the extracted features from the pre-trained CNN on the disk to later train the SVMs. This requires hundreds of gigabytes of storage.
1. R-CNN depends on the Selective Search algorithm for generating region proposals, which takes a lot of time. Moreover, this algorithm cannot be customized to the detection problem.
1. Each region proposal is fed independently to the CNN for feature extraction. This makes it impossible to run R-CNN in real-time.


### Fast R-CNN} 
#### Insight 1: RoI (Region of Interest) Pooling}
For the forward pass of the CNN, Girshick realized that for each image, a lot of proposed regions for the image invariably overlapped causing us to run the same CNN computation again and again (~2000 times!). His insight was simple — Why not run the CNN just once per image and then find a way to share that computation across the ~2000 proposals?
#### Contributions}

1. Proposed a new layer called ROI Pooling that extracts equal-length feature vectors from all proposals (i.e. ROIs) in the same image.
1. Fast R-CNN does not cache the extracted features and thus does not need so much disk storage compared to R-CNN, which needs hundreds of gigabytes.
1. Training is single-stage, using a multi-task loss.
1. Fast R-CNN is more accurate than R-CNN

\begin{figure*}
    \centering
        \includegraphics [width=0.7\textwidth, height = 0.4\textheight]{Diagrams/Faster-RCNN.png}
    \caption{Pipeline of Faster-RCNN}
    \label{fig:fastrcnn}
\end{figure*}
See fig \ref{fig:fastrcnn}
#### Drawback}
Still generating those 2000 region proposals is done using selective search method which is in-efficient. This leads to the idea of developing Faster-RCNN which extracts the region proposals using CNNs(Region Proposal Network).  
### Faster-RCNN}
#### Contributions}

1. Proposing region proposal network (RPN) which is a fully convolutional network that generates proposals with various scales and aspect ratios. The RPN implements the terminology of neural network with attention to tell the object detection (Fast R-CNN) where to look.
1. Rather than using pyramids of images (i.e. multiple instances of the image but at different scales) or pyramids of filters (i.e. multiple filters with different sizes), this paper introduced the concept of anchor boxes. An anchor box is a reference box of a specific scale and aspect ratio. With multiple reference anchor boxes, then multiple scales and aspect ratios exist for the single region. This can be thought of as a pyramid of reference anchor boxes. Each region is then mapped to each reference anchor box, and thus detecting objects at different scales and aspect ratios.
1. The convolutional computations are shared across the RPN and the Fast R-CNN. This reduces the computational time.

#### Pipeline}

1. The RPN generates region proposals.
1. For all region proposals in the image, a fixed-length feature vector is extracted from each region using the ROI Pooling layer.
1. The extracted feature vectors are then classified using the Fast R-CNN.
1. The class scores of the detected objects in addition to their bounding-boxes are returned.

# CNN Interpretability Techniques}
### Prerequisites}
#### Interpretability vs Explainability}

1. Interpretability :  Interpretability is about the extent to which a cause and effect can be observed within a system. Or, to put it another way, it is the extent to which you are able to predict what is going to happen, given a change in input or algorithmic parameters. It’s being able to look at an algorithm and go yep, I can see what’s happening here.

1. Explainability : Explainability, meanwhile, is the extent to which the internal mechanics of a machine or deep learning system can be explained in human terms. 

1. Interpretability is about being able to discern the mechanics without necessarily knowing why. Explainability is being able to quite literally explain what is happening.


#### Why Interpretability ?}
We must build ‘transparent’ models that explain why they predict what they predict. 

1. First, when the AI is relatively weaker than the human and not yet reliably ‘deployable’, the goal of transparency and explanations is to identify the failure mode.
1. Second, when the AI is on par with humans and reliably ‘deployable’, the goal is to establish appropriate trust and confidence in users.
1. Third, when the AI is significantly stronger than humans, the goal of the explanations is in machine teaching i.e teaching humans how to take better decisions.
 

### Class Activation Maps (CAM)}
\begin{figure*}
    \centering
        \includegraphics [width=1\textwidth, height = 0.3\textheight]{Diagrams/CAM.png}
    \caption{}
    \label{fig:cam}
\end{figure*}
Video to be watched : \url{https://www.youtube.com/watch?v=vTY58-51XZA} \newline
Github : \url{https://github.com/nickbiso/Keras-Class-Activation-Map}

1. A Class Activation map for a particular category indicates the discriminative region used by CNN to identify the category.
1. The Network mainly consists of a large number of convolutional layers and just before the final output layer, we perform Global Average Pooling. The features thus obtained are fed to a fully connected layer having with softmax activation which produces the desired output. We can identify the importance of the image regions by projecting back the weights of the output layer on the convolutional feature maps obtained from the last Convolution Layer. This technique is known as Class Activation Mapping.
1. The Global Average Pooling layer(GAP) is preferred over the Global MaxPooling Layer(GMP) because GAP layers help to identify the complete extent of the object as compared to GMP layer which identifies just one discriminative part. This is because in GAP we take an average across all the activation which helps to find all the discriminative regions while GMP layer just considers only the most discriminative one.
1. The weights of the final layer corresponding to that class are extracted. Also, the feature map from the last convolutional layer is extracted.
1. Finally, the dot product of the extracted weights from the final layer and the feature map is calculated to produce the class activation map. The class activation map is upsampled by using Bi-Linear Interpolation and superimposed on the input image to show the regions which the CNN model is looking at. Refer fig \ref{fig:cam}
1. Drawbacks : It requires feature maps to directly precede the softmax layers, so it is applicable to a particular kind of CNN architectures that perform global average pooling over convolutional maps immediately before prediction. (i.e conv feature maps $\rightarrow$ global average pooling $\rightarrow$ softmax layer).



### Grad-CAM}
Check out demo : \url{http://gradcam.cloudcv.org/} \newline
Watch the demo : \url{https://www.youtube.com/watch?v=COjUB9Izk6E}
#### Applications}
Grad-CAM is applicable to a wide variety of CNN model-families: 

    1. CNNs with fully-connected layers (e.g. VGG)
    1. CNNs used for structured outputs (e.g. captioning)
    1. CNNs used in tasks with multi-modal inputs (e.g. visual question an- swering) or reinforcement learning, without architectural changes or re-training.


#### Introduction}

1. Consider image classification – a ‘good’ visual explanation from the model for justifying any target category should be (a) class- discriminative (i.e. localize the category in the image) and (b) high-resolution (i.e. capture fine-grained detail)
1. Past work : Guided Back- propagation and Deconvolution are high-resolution and highlight fine-grained details in the image, but are not class-discriminative.
1. In contrast, localization approaches like CAM or our pro-posed method Gradient-weighted Class Activation Mapping (Grad-CAM), are highly class-discriminative.

1. Going from deep to shallow layers, the discriminative ability of Grad-CAM significantly reduces as we encounter layers with different output dimensionality and also they lack the ability to show fine-grained importance like pixel-space gradient visualization methods (Guided Back- propagation and Deconvolution).

1. It is possible to fuse existing pixel-space gradient visualizations with Grad-CAM to create Guided Grad-CAM visualizations that are both high-resolution and class discriminative 

1. Grad-Cam, unlike CAM, uses the gradient information flowing into the last convolutional layer of the CNN to understand each neuron for a decision of interest because last convolutional layers are expected to have the best compromise between high-level semantics and detailed spatial information.


### Problem with Gradients}
Blog : \url{https://towardsdatascience.com/interpretable-neural-networks-45ac8aa91411}

1. Possibly the most intepretable model — and therefore the one we will use as inspiration — is a regression. In a regression, each feature x is assigned some weight, w, which directly tells me that feature’s importance to the model.
1. Specifically, for the ith feature of a specific data point, the feature’s contribution to the model output is $w_i \times x_i$. What does this weight w represent? Well, since a regression is
     $$Y = w_1 \times x_1 + w_2 \times x_2 + ........+ w_n \times x_n + b$$
Then, 
    $$w_i = \frac{\partial Y}{\partial x_i}$$
1. In other words, the weight assigned to the $i^{th}$ feature tells us the gradient of that feature with respect to the model’s prediction: how the model’s prediction changes as the feature changes.
Conveniently, this gradient is easy to calculate for neural networks. So, in the same way that for a regression, a feature’s contribution is
            $$w_i \times x_i = x_i \times \frac{\partial Y}{\partial x_i}$$
, perhaps the gradient can be used to explain the output of a neural network.
1. There are two issues we run into when trying to use this approach:
Firstly, feature importances are relative. For gradient boosted decision trees, a feature’s shap value tells me how a feature changed the prediction of the model relative to the model not seeing that feature. Since neural networks can’t handle null input features, we’ll need to redefine a feature’s impact relative to something else.\par
To overcome this, we’ll define a new baseline: what am I comparing my input against? One example is for the MNIST digits dataset. Since all the digits are white, against a black background, perhaps a reasonable background would be a fully black image, since this represents no information about the digits. Choosing a background for other datasets is much less trivial — for instance, what should the background be for the ImageNet dataset? We’ll discuss a solution for this later, but for now let’s assume that we can find a baseline for each dataset.\par
The second issue is that using the gradient of the output with respect to the input works well for a linear model — such a regression — but quickly falls apart for nonlinear models. To see why, let’s consider a “neural network” consisting only of a ReLU activation, with a baseline input of x=2.
Now, lets consider a second data point, at x = -2. ReLU(x=2) = 2, and ReLU(x=-2) = 0, so my input feature x = -2 has changed the output of my model by 2 compared to the baseline. This change in the output of my model has to be attributed to the change in x, since its the only input feature to this model, but the gradient of ReLU(x) at the point x = -2 is 0! This tells me the contribution of x to the output is 0, which is obviously a contradiction.
1. This happened for two reasons: firstly, we care about a finite difference in the function (the difference between the function when x = 2 and when x = -2), but gradients calculate infinitesimal differences. Secondly, the ReLU function can get saturated — once x is smaller than 0, it doesn’t matter how much smaller it gets, since the function will only output 0. As we saw above, this results in inconsistent model intepretations, where the output changes with respect to the baseline, but no features are labelled as having caused this change. 
1. These inconsistencies are what Integrated Gradients and DeepLIFT attempt to tackle. They both do this by recognizing that ultimately, what we care about is not the gradient at the point x; we care about how the output changed from the baseline as the input changed from the baseline.

### Fundamental axioms which needs to be satisfied to be a attribution method}
Axiom :

    1. Sensitivity(a) : An attribution method satisfies Sensitivity(a) if for every
    input and baseline that differ in one feature but have different predictions then the differing feature should be given a non-zero attribution. 
    1. Implementation Invariance : Two networks are functionally equivalent if their outputs are equal for all inputs, despite having very different implementations. Attribution methods should satisfy Implementation Invariance, i.e., the attributions are always identical for two functionally equivalent networks. 

If an attribution method fails to satisfy the above axioms, the attributions are potentially sensitive to unimportant aspects of the models. For
#### Gradients}

    1. Gradients (of the output with respect to the input) is a natural analog of the model coefficients for a deep network, and therefore the product of the gradient and feature values is a reasonable starting point for an attribution method.
    1. Gradients violate Sensitivity(a): For a concrete example, consider a one variable, one ReLU network, f(x) = 1 - ReLU(1-x). Suppose the baseline is x = 0 and the input is x = 2. The function changes from 0 to 1, but because $f$ becomes flat at x = 1, the gradient method gives attribution of 0 to x. Intuitively, gradients break Sensitivity because the prediction function may flatten at the input and thus have zero gradient despite the function value at the input being different from that at the baseline.
 
### Integrated Gradients}
Check out the best blog and read directly from it. \textbf{\url{https://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html}}
#### Introduction}

1. The approach taken by integrated gradients is to ask the following question: what is something I can calculate which is an analogy to the gradient which also acknowledges the presence of a baseline?
1. Part of the problem is that the gradient (of the output relative to the input) at the baseline is going to be different then the gradient at the output I measure; ideally, I would consider the gradient at both points. This is not enough though: consider a sigmoid function, where my baseline output is close to 0, and my target output is close to 1:
\begin{figure*}
    \centering
        \includegraphics [width=1\textwidth, height = 0.25\textheight]{Diagrams/sigmoid_ig.png}
    % \caption{}
    \label{fig:selfattention}
\end{figure*}
1. Here in fig \ref{fig:sigmoid_ig}, both the gradient at my baseline and at my data point are going to be close to 0. In fact, all the interesting — and informative — gradients are in between the two data points, so ideally we would find a way to capture all of that information too.
1. That’s exactly what Integrated Gradients do, by calculating the integral of the gradients between the baseline and the point of interest. Actually calculating the integral of the gradients is intractable, so instead, they are approximated using a Reimann sum: the gradients are taken at lots of small steps between the baseline and the point of interest.

#### Algorithm}

1. Start from the baseline where baseline can be a black image whose pixel values are all zero or an all-white image, or a random image. Baseline input is one where the prediction is neutral and is central to any explanation method and visualizing pixel feature importances.
1. Generate a linear interpolation between the baseline and the original image. Interpolated images are small steps$(\alpha)$ in the feature space between your baseline and input image and consistently increases with each interpolated image’s intensity.
$$\gamma(\alpha) = x' + \alpha \times (x - x')$$  for $\alpha \in [0, 1]$
\newline where 
$\gamma(\alpha)$ is the interpolated image \newline
$x'$ -- Baseline image and 
$x$ -- Input image
1. Calculate gradients to measure the relationship between changes to a feature and changes in the model’s predictions.
1. Compute the numerical approximation through averaging gradients.
1. Scale IG to the input image to ensure that the attribution values are accumulated across multiple interpolated images are all in the same units. Represent the IG on the input image with the pixel importances.
1. Tensorflow implementation - \url{https://www.tensorflow.org/tutorials/interpretability/integrated_gradients}

### How to pick a baseline}

    1. As promised, we will now return to picking baseline. Aside from some very obvious cases (eg. the MNIST example above), deciding what the baseline inputs are is extremely non trivial, and might require domain expertise.
1. An alternative to manually picking a baseline is to consider what the prior distribution of a trained model is. This can give us a good idea of what the model is thinking when it has no information at all.
1. For instance, if I have trained a model on the ImageNet, what is its prior assumption going to be that a new photo it sees if of a meerkat? If $2\%$ of the photos in the ImageNet dataset are of a meerkat, then the model is going to think that the new photo it sees has a $2\%$ chance of being a meerkat. It will then adjust its prediction accordingly when it actually sees the photo. It makes sense to measure the impact of the inputs relative to this prior assumption.
1. So how can I pick a baseline which is $2\%$ meerkat? Well, a good approach might be to take the mean of the dataset, simply by averaging the images in the dataset together. This is the approach used in the shap library’s implementations of Integrated Gradients and DeepLIFT. Conveniently, it removes the need to be a domain expert to pick an appropriate baseline for the model being interpreted.