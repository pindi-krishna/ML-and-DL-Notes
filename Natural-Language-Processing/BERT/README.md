# Bert

## Introduction
BERT is basically a trained Transformer Encoder stack.

The paper presents two model sizes for BERT:

1. BERT BASE – Comparable in size to the OpenAI Transformer in order to compare performance 
    1. 12 Stacked Encoder
    1. 768 units in Feed Forward Layer
    1. 12 Attention Heads

1. BERT LARGE – A ridiculously huge model which achieved the state of the art results reported in the paper 
    1. 24 Stacked Encoder
    1. 1024 units in Feed Forward Layer
    1. 16 Attention Heads

There is no decoder in BERT. It is used for all the encoding tasks. 

## Sentence Classification Task

1. The first input token is supplied with a special [CLS] token for reasons that will become apparent later on. CLS here stands for Classification.

1. Just like the vanilla encoder of the transformer, BERT takes a sequence of words as input which keep flowing up the stack. Each layer applies self-attention, and passes its results through a feed-forward network, and then hands it off to the next encoder.

1. Each position outputs a vector of size hidden_size (768 in BERT Base). For the sentence classification example we’ve looked at above, we focus on the output of only the first position (that we passed the special [CLS] token to).

1. That vector can now be used as the input for a classifier of our choosing. The paper achieves great results by just using a single-layer neural network as the classifier.

![bert](./Images/bert-sentence-cls.png)
