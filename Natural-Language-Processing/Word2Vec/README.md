# Word to Vector (word2vec)

[Source](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

### Introduction

1. A Word Embedding format generally tries to map a word using a dictionary to a vector.
1. Word Embeddings are Word converted into numbers.
1. \textbf{Why} :  Many Machine Learning algorithms and almost all Deep Learning Architectures are incapable of processing strings or plain text in their raw form. They require numbers as inputs to perform any sort of job. Therefore, We do this using different word embeddings. 
1. Simple One hot representations doesn't work well because context in which that particular occurs is not taken into consideration. A quote says that "You shall know a word by the company it keeps". In english vocabulary, we can exactly understand a word only when we know its context. 


#### Types

1. Frequency based Embedding

    1. Count Vector
    1. TF-IDF Vector
    1. Co-Occurrence Vector

1. Prediction based Embedding

    1. CBOW(Continuous bag of words) 
    1. Skip-gram model


### Frequency based Embedding
#### Count Vector

1. Consider a Corpus C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens will form our dictionary and the size of the Count Vector matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
1. **Limitation** : In real world applications we might have a corpus which contains millions of documents. And with millions of document, we can extract hundreds of millions of unique words. So basically, the matrix that will be prepared like above will be a very sparse one and inefficient for any computation.

#### TF-IDF vectorization

1. Common words like ‘is’, ‘the’, ‘a’ etc. tend to appear quite frequently in comparison to the words which are important to a document. For example, a document A on Lionel Messi is going to contain more occurences of the word “Messi” in comparison to other documents. But common words like “the” etc. are also going to be present in higher frequency in almost every document.
1. Ideally, what we would want is to down weight the common words occurring in almost all documents and give more importance to words that appear in a subset of documents.
1. TF-IDF works by penalising these common words by assigning them lower weights while giving importance to words like Messi in a particular document.

|Document 1 |  Document 1  | Document 2  | Document 2|
|------|-------|-------|-------|
|Token | Count | Token | Count |
| This | 1 |  This | 1 |
| is | 1  | is | 2 |
| about | 2  | about |1 |
| Messi | 4 | TF-IDF | 1 |

$$TF = \frac{\text{Number of times term t appears in a document}}{\text{Number of terms in the document}}$$

So, $$TF(This,Document1) = 1/8$$

$$TF(This, Document2)=1/5$$

1. It denotes the contribution of the word to the document i.e words relevant to the document should be frequent. eg: A document about Messi should contain the word ‘Messi’ in large number.

1. IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.

where N is the number of documents and n is the number of documents a term t has appeared in.

So, $$ IDF(This) = log(2/2) = 0.$$

1. So, how do we explain the reasoning behind IDF? Ideally, if a word has appeared in all the document, then probably that word is not relevant to a particular document. But if it has appeared in a subset of documents then probably the word is of some relevance to the documents it is present in.

1. Let us compute IDF for the word ‘Messi’.

$$IDF(Messi) = log(2/1) = 0.301.$$

Now, let us compare the TF-IDF for a common word ‘This’ and a word ‘Messi’ which seems to be of relevance to Document 1.

$$ TF-IDF(This,Document1) = (1/8) * (0) = 0 $$

$$ TF-IDF(This, Document2) = (1/5) * (0) = 0 $$

$$ TF-IDF(Messi, Document1) = (4/8)*0.301 = 0.15 $$

As, you can see for Document1 , TF-IDF method heavily penalises the word ‘This’ but assigns greater weight to ‘Messi’. So, this may be understood as ‘Messi’ is an important word for Document1 from the context of the entire corpus.

### Prediction based Embedding
1. To encode the probability of a word given its context.  
[Stanford-Notes](https://cs224d.stanford.edu/lecture_notes/notes1.pdf)

#### CBOW

1. One approach is to treat {"The", "cat", ’over", "the’, "puddle"} as a
context and from these words, be able to predict or generate the center word "jumped". This type of model we call a Continuous Bag of Words (CBOW) Model.
1. Notation for CBOW Model:
    1. $w_i$ : Word i from vocabulary V
    1. $V \in \Re^{n\times|V|} $ : Input word matrix
    1. $v_i$ : $i_{th}$ column of V, the input vector
    representation of word wi
    1. $U \in R^{n \times |V|}$ : Output word matrix
    1. $u_i$ : $i_{th}$ column of U, the output vector
    representation of word ${w_i}$.

1. Algorithm :
    
    1. We breakdown the way this model works in these steps: 
    1. We generate our one hot word vectors $(x^{(c-m)}, . . . , x^{(c-1)}, x^{(c+1)}, . . . , x^{(c+m)})$
    for the input context of size m.
    2. We get our embedded word vectors for the context $$(v_{c-m} = Vx^{(c-m)}
    , v_{c-m+1} = Vx^{(c-m+1)}, . . ., v_{c+m} = Vx^{(c+m)})$$
    1. Average these vectors to get 
    $$v^` = \frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}$$
    1. Generate a score vector $z = Uv^`$
    1. Turn the scores into probabilities 
    $y^` = softmax(z)$
    1. We desire our probabilities generated, $y^`$, to match the true probabilities, y, which also happens to be the one hot vector of the
    actual word.
    
1. So now that we have an understanding of how our model would work if we had a V and U, how would we learn these two matrices?
1. Well, we need to create an objective function. Very often when we are trying to learn a probability from some true probability, we look to information theory to give us a measure of the distance between
two distributions. Here, we use a popular choice of distance/loss measure, cross entropy $H(y^`, y)$.

1. The intuition for the use of cross-entropy in the discrete case can be derived from the formulation of the loss function: $$H(y^`, y) = −|V|\sum_{j=1} y_j log({y^`}_j)$$

1. Let us concern ourselves with the case at hand, which is that y is a one-hot vector. Thus we know that the above loss simplifies to simply: $$H(y^`, y) = −y_i log({y^`}_i)$$

1. In this formulation, c is the index where the correct word’s one hot vector is 1. We can now consider the case where our prediction was perfect and thus ${y^`}_c = 1$. We can then calculate $H(y^`, y) = -1 log(1) = 0$. 

1. Thus, for a perfect prediction, we face no penalty or loss. Now let us consider the opposite case where our prediction was very bad and thus $yˆc = 0.01$. As before, we can calculate our loss to be $H(y^`, y) = -1 log(0.01) ≈ 4.605$. 

1. We can thus see that for probability distributions, cross entropy provides us with a good measure of distance.

1. We thus formulate our optimization objective as: $$minimize J = - log P(w_c|w_{c-m}, . . . , w_{c-1}, w_{c+1}, . .w_{c+m})$$

1. We use gradient descent to update all relevant word vectors $u_c$ and $v_j$.


#### Skip Gram Model
Predicting surrounding context words given a center word.

1. Another approach is to create a model such that given the center word "jumped", the model will be able to predict or generate the surrounding words "The", "cat", "over", "the", "puddle". Here we call the word "jumped" the context. We call this type of model a SkipGram model. 
1. Let’s discuss the Skip-Gram model above. The setup is largely the same but we essentially swap our x and y i.e. x in the CBOW are now y and vice-versa. The input one hot vector (center word) we will represent with an x (since there is only one). And the output vectors
as $y_{(j)}$. We define V and U the same as in CBOW.
1. Algorithm : 
    
    1. We generate our one hot input vector x
    1. We get our embedded word vectors for the context $v_c = Vx$
    1. Since there is no averaging, just set $v^` = v_c$ ?
    1. Generate 2m score vectors, $u_{c-m}, . . . , u_{c-1}, u_{c+1}, . . . , u_{c+m}$ using $u = Uv_c$
    1. Turn each of the scores into probabilities, y = softmax(u)
    1. We desire our probability vector generated to match the true probabilities which is $y_{(c-m)} , . . . , y_{(c-1)}, y_{(c+1)}, . . . , y_{(c+m)}$
    , the one hot vectors of the actual output.
    
1. As in CBOW, we need to generate an objective function for us to evaluate the model. A key difference here is that we invoke a Naive Bayes assumption to break out the probabilities. If you have not seen this before, then simply put, it is a strong (naive) conditional
independence assumption. In other words, given the center word, all output words are completely independent.
$$minimize J = - log P(w_{c-m}, . . . , w_{c-1}, w_{c+1}, . . . , w_{c+m}|w_c)$$



### Glove representation
1. For Implementation, Check out this [Blog](https://towardsdatascience.com/a-comprehensive-python-implementation-of-glove-c94257c2813d).
2. Main Source is Glove Stanford Paper 
3. Mithesh sir IIT Madras NPTEL DL videos.  

#### Introduction
The two main model families for learning wordvectors are: 

1. Global matrix factorization methods, such as latent semantic analysis (LSA).
1. Local context window methods, such as the skip-gram model of Mikolovet al.  

Currently, both families suffer significant drawbacks.  While methods like LSA efficiently  leverage  statistical  information,  they  do relatively poorly on the word analogy task,  indicating a sub-optimal vector space structure. Methods like skip-gram may do better on the analogy task, but they poorly utilize the statistics of the cor-pus since they train on separate local context windows instead of on global co-occurrence counts. \par

**Idea: Why not combine them ?**
They proposed a specific weighted least squares model that trains on global word-word co-occurrence counts and thus makes efficient use of statistics.

#### Algorithm

1. X = PPMI co-occurence matrix. 
1. $X_{ij}$ represents the number of times word $i$ and word $j$ occur together. 
1. $w_i$ :  Vector representation of word $i$
1. $w_j$ :  Vector representation of word $i$
1. $b_i$ : Bias of word $i$ 
1. $b_j$ : Bias of word $j$ 

Optimization problem is formulated as follows : 
$$L = ({w_i}^Tw_j + b_i + b_j - log(X_{ij})^2$$
We are trying to get vector representations of word $i$ and word $j$ closer to the $X_{ij}$. 