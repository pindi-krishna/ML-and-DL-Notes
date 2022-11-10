# Generative Pre-trained Transformer (GPT) 

1. [OpenAI Blog](https://openai.com/blog/language-unsupervised/)
1. [Jay Alammar Blog](https://jalammar.github.io/illustrated-gpt2/)

Unsupervised learning served as pre-training objective for supervised fine-tuned models, hence the name Generative Pre-training.

## Introduction

1. Language model is – basically a machine learning model that is able to look at part of a sentence and predict the next word.

2. In this sense, we can say that the GPT is basically the next word prediction feature of a keyboard app, but one that is much larger and more sophisticated than what your phone has.  

3. This system works in two stages; 
    1. Train a transformer model on a very large amount of data in an unsupervised manner — using language modeling as a training signal 

    2. Then fine-tune this model on much smaller supervised datasets to help it solve specific tasks.

4. **Why unsupervised learning** ? 
    
    Since  it removes the bottleneck of explicit human labeling and also scales well with current trends of increasing compute and availability of raw data. 

## Applications 

GPT models can perform various NLP tasks listed below without any supervised training. 
1. Question answering
1. Textual entailment
1. Text summarisation 

## GPT-1


## GPT-2

1. The GPT-2 was trained on a massive 40GB dataset called WebText that the OpenAI researchers crawled from the internet as part of the research effort.

1. GPT-2 with model sizes are available. One of the main distinguishing factors between the different GPT2 model sizes:
    | Size| Number of Decoders | Model Dimensionality |
    -------|---------------|-------------|
    | Small | 12 | 768 |
    | Medium | 24 | 1024 |
    | Large | 36 | 1280 |
    | Extra Large | 48 | 1600 | 
    
1. The GPT-2 is built using transformer decoder blocks. It outputs one token at a time. The way these models actually work is that after each token is produced, that token is added to the sequence of inputs. And that new sequence becomes the input to the model in its next step. This is an idea called “auto-regression”. 

1. One key difference in the self-attention layer in Decoder, is that it masks future tokens – not by changing the word to [mask] like BERT, but by interfering in the self-attention calculation blocking information from tokens that are to the right of the position being calculated.
 



