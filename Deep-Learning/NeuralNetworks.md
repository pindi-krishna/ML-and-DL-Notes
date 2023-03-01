# Neural Networks

## Neuron
Single neuron == linear regression without applying activation(perceptron). Followed by an activation function gives rise to linear/logistic regression. 

## Neural Network

**Simple Neural Network**

Each Input will be connected to the hidden layer and the NN will decide the connections. As the dataset size increases, NN performance increases.  

![NN](./Images/NN.jpg)

Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

![NNR](./Images/NNR.png)

In above figure ,Let $n_H$ = \# Hidden neurons, $n_X$ = \# input features, $n_y$ = \# output Classes.

1. Variables :
    
    1. $W_1 : (n_H, n_x) $, $b_1 : (n_H, 1) $
    1. $Z_1 : (n_H, 1) $, $a_1 : (n_H, 1) $
    1. $W_2 : (n_y, n_H)$, $b_2 : (n_y, 1) $
    1. $Z_2 : (n_y, 1) $,$a_2 : (n_y, 1) $
    
1. Forward Propagation : 
    
    1. $Z_1 = W_1 X + b_1 $
    1. $A_1 = g_1(Z_1)$
    1. $Z_2 = W_2 A_1 + b_2$
    1. $A_2 = Sigmoid(Z_2)$
1. Compute Cross Entropy (CE) Loss 
    $$L(A2, Y) = -Ylog(A2) - (1-Y)log(1-A2)$$    
1. Backward Propagation : Element wise product ($\times$)
    
    1. $dZ_2 = A2-Y$ 
    1. $dW_2 = \frac{1}{m}(dZ_2 \times {A_1}^T)$
    1. $db_2 = \frac{1}{m}\sum dZ_2$
    1. $dZ_1 = ({W_2}^T \times dZ_2) \times {g'}_1(Z_1) $  
    1. $dW_1 = \frac{1}{m} (d Z_1 \times X^T)$  
    1. $db_1 = \frac{1}{m}\sum dZ_1$
    

## Deep L-layer neural network

1. Shallow NN is a NN with one or two layers whereas Deep NN has three or more layers.

1. Variables :
    
    1. $n[l]$ is the number of neurons in a specific layer l. 
        1. $n[0]$ denotes the number of neurons input layer. 
        1. $n[L]$ denotes  the number of neurons in output layer.
    1. $g[l]$ is the activation function in $l$ layer.
    $a[l] = g[l](z[l])$.
    1. $w[l]$ weights is used for $z[l]$, shape : $(n[l], n[l-1])$
    1. $b[l]$ bias for layer $l$, shape : $(n[l],1)$
    1. $x = a[0]$ and  $y' = a[L]$ 
    
1. Forward Propagation : 
    $$Input  = A[l-1]$$
    $$ Z[l] = {W[l]}^T.A[l-1] + b[l]$$
    $$ A[l] = g[l](Z[l])$$
    $$ Output =  A[l]$$

1. Loss Function :  We generally use log-loss for classification.
   $$L = -y log(A) - (1-y)log(1-A) $$
   $$dA[L] = \frac{-y}{a}+\frac{1-y}{1-a} $$
   where $dA[L] = \frac{\partial L }{\partial A[L]}$
1. Backward Propagation : 
    $$Input =  da[l] $$
    $$dZ[l] = dA[l].g'[l](Z[l]) $$
    $$ dW[l] = \frac{1}{m} (dZ[l]{A[l-1]}^T)$$
    $$ db[l] = \frac{1}{m} sum(dZ[l])$$
    Dont forget $axis=1$ and $keepdims=True$ in the code.
    $$ Output = dA[l-1] = {w[l]}^T. dZ[l]$$ 
    
1. Algorithm :
    
    1. Initialize weights randomly
    1. Forward Propagation
    1. Compute Cost function.
    1. Compute gradients using Back Propagation
    1. Update all weights 
    1. Repeat above steps $ii-v$ till the weights converge (i.e.) till the lowest loss is achieved. 
    
1. Learnable Parameters : $W$ and $b$.
1. Hyper parameters (parameters that control the algorithm):
       
    1. Learning rate.
    1. Batch size
    1. Number of iteration.
    1. Number of hidden layers L.
    1. Number of hidden units n.
    1. Choice of activation functions.
    
    You have to experiment with different values of hyper parameters to find the best combination.

## Vanishing / Exploding gradients

1.  The Vanishing / Exploding gradients occurs when your derivatives become very small or very big.
1. Illustration : 
Suppose  that we have a deep neural network with number of layers $L$, and all the activation functions are linear and each $b = 0$. Then  
    $$ Y' = W[L]W[L-1].....W[2]W[1]X$$ 
1. Exploding Gradients :  If $W > I$ (Identity matrix) the activation and gradients will explode.
    $$W[l] = 
    \begin{bmatrix} 
    1.5 & 0 \\  
    0 & 1.5 \\ 
    \end{bmatrix}
    $$ 
    ($l != L$ because of different dimensions in the output layer)
    $$Y' = W[L]
    \begin{bmatrix}
    1.5 & 0 \\
    0 & 1.5 \\
    \end{bmatrix}^{L-1} X = 1.5^L$$ 
    which will be very large.
1. Vanishing Gradients : If W < I (Identity matrix) the activation and gradients will vanish.
    $$ W[l] = 
    \begin{bmatrix}
    0.5 & 0 \\
    0 & 0.5 \\
    \end{bmatrix}
    $$

    $$Y' = W[L]
    \begin{bmatrix}
        0.5 & 0 \\
        0 & 0.5 \\
        \end{bmatrix}^{L-1} 
        X = 0.5^L $$  
    which will be very small.
    
1. Recently Microsoft trained 152 layers (ResNet), which is a really giant network. With such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller, then gradient descent will take tiny steps. The model may learn something in a very long time or end up learning nothing if graidents tends to zero.

1. There is a partial solution that doesn't completely solve this problem but it helps a lot - careful choice of how you initialize the weights.