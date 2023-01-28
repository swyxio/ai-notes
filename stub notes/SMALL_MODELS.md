## opinions

large models are demoralizing https://www.reddit.com/r/MachineLearning/comments/wiqjxv/d_the_current_and_future_state_of_aiml_is/

## history

- Computer Vision practitioners will remember when [SqueezeNet](https://arxiv.org/abs/1602.07360) came out in 2017, achieving a 50x reduction in model size compared to [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), while meeting or exceeding its accuracy.
- [DistilBERT](https://arxiv.org/abs/1910.01108) is perhaps its most widely known achievement. Compared to the original BERT model, it retains 97% of language understanding while being 40% smaller and 60% faster. You can try it [here](https://huggingface.co/distilbert-base-uncased). The same approach has been applied to other models, such as Facebook's [BART](https://arxiv.org/abs/1910.13461), and you can try DistilBART [here](https://huggingface.co/models?search=distilbart).
	- https://twitter.com/danielgross/status/1619417503561818112?s=46&t=swb0mG9U2bNJt3tlgnWOyw running on apple m2 pro at 50% of an a100 lol
-  [Big Science](https://bigscience.huggingface.co/) project are also very impressive. As visible in this graph included in the [research paper](https://arxiv.org/abs/2110.08207), their T0 model outperforms GPT-3 on many tasks while being 16x smaller.

## directions


CRAMMING: TRAINING A LANGUAGE MODEL ON A SINGLE GPU IN ONE DAY https://arxiv.org/pdf/2212.14034v1.pdf


light BERT model?
https://twitter.com/deepwhitman/status/1610354790772637697?s=46&t=m_H3h2AbaafHjQJEu1cHJw


https://arxiv.org/abs/2301.00774
Massive Language Models Can Be Accurately Pruned in One-Shot

https://arxiv.org/abs/2202.08906 [**ST-MoE: Designing Stable and Transferable Sparse Expert Models**](https://arxiv.org/abs/2202.08906)

Best sparse model paper. Meticulous experiments, comprehensive evals and the goto reference for all sparse modeling


H3 state space models https://twitter.com/mathemagic1an/status/1617620133182332928?s=46&t=I3QXnpdfF5CXpszIYkv4pQ