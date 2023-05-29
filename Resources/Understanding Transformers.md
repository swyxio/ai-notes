
2017 paper
- https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html?m=1



-  LSTM is dead. Long Live Transformers! ([talk](https://www.youtube.com/watch?v=S27pHKBEp30))
	-  the evolution of natural language processing (NLP) techniques, starting with the limitations of bag-of-words models and vanilla recurrent neural networks (RNNs), which suffer from vanishing and exploding gradients. 
	- The introduction of long short-term memory (LSTM) resolved these issues but was still challenging to train and lacked transfer learning abilities, leading to the development of the transformer model. 
	- The transformer model uses **self-attention** and **a feedforward neural network** to read input sequences and create output sequences, incorporating **multi-headed attention** to generate multiple attention outputs with different sets of parameters. 
	- Key innovations of the transformer model include **positional encoding** and the use of **ReLU activation** functions. 
	- The speaker highlights the advantages of using transformers and models like Roberta for training models on large-scale unsupervised text data, enabling transfer learning and reduced training time and resources.
	- Despite being replaced in most areas by transformers, LSTM still has applications in real-time control. 
	- The speaker also compares word CNN to transformers and observes that transformers can offer contextual answers more efficiently across the entire document.
- Little Book of Deep Learning ([pdf](https://fleuret.org/public/lbdl.pdf))
	- The notion of layer
	- Linear layer
	- Activation functions
	- Pooling
	- Dropout
	- Normalizing layers
	- Skip connections
	- Attention layers
	- Token embedding
	- Positional encoding
	- Architectures
	- Multi-Layer Perceptrons
	- Convolutional networks
	- Attention models
- Reminder that my deep learning course [@unige_en](https://twitter.com/unige_en)is entirely available on-line. 1000+ slides, ~20h of screen-casts. [https://fleuret.org/dlc/](https://t.co/6OVyjPdwrC)
- https://e2eml.school/transformers.html Transformers from Scratch


https://news.ycombinator.com/item?id=35712334
The Illustrated Transformer is fantastic, but I would suggest that those going into it really should read the previous articles in the series to get a foundation to understand it more, plus later articles that go into GPT and BERT, here's the list:

A Visual and Interactive Guide to the Basics of Neural Networks - [https://jalammar.github.io/visual-interactive-guide-basics-n...](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)

A Visual And Interactive Look at Basic Neural Network Math - [https://jalammar.github.io/feedforward-neural-networks-visua...](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/)

Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) - [https://jalammar.github.io/visualizing-neural-machine-transl...](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

The Illustrated Transformer - [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) - [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)

The Illustrated GPT-2 (Visualizing Transformer Language Models) - [https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)

How GPT3 Works - Visualizations and Animations - [https://jalammar.github.io/how-gpt3-works-visualizations-ani...](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

The Illustrated Retrieval Transformer - [https://jalammar.github.io/illustrated-retrieval-transformer...](https://jalammar.github.io/illustrated-retrieval-transformer/)

The Illustrated Stable Diffusion - [https://jalammar.github.io/illustrated-stable-diffusion/](https://jalammar.github.io/illustrated-stable-diffusion/)

If you want to learn how to code them, this book is great: [https://d2l.ai/chapter_attention-mechanisms-and-transformers...](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)

Transformer Taxonomy (the last lit review)
https://kipp.ly/blog/transformer-taxonomy/
- It covers 22 models, 11 architectural changes, 7 post-pre-training techniques and 3 training techniques (and 5 things that are none of the above).

attention visualization https://catherinesyeh.github.io/attn-docs/

## flash attention

https://twitter.com/amanrsanger/status/1657835933503479808?s=46&t=90xQ8sGy63D2OtiaoGJuww