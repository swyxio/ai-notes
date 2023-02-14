## opinions

large models are demoralizing https://www.reddit.com/r/MachineLearning/comments/wiqjxv/d_the_current_and_future_state_of_aiml_is/

According to Alex Graveley, one of the creators of Github’s Copilot, there is a 1% drop in completions for every additional 10ms of latency. https://www.buildt.ai/blog/incorrectusage

naveen rao https://twitter.com/NaveenGRao/status/1625176665746964480 Smaller models w/ some clever precomputing of results is needed.

## history

- Computer Vision practitioners will remember when [SqueezeNet](https://arxiv.org/abs/1602.07360) came out in 2017, achieving a 50x reduction in model size compared to [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), while meeting or exceeding its accuracy.
- [DistilBERT](https://arxiv.org/abs/1910.01108) is perhaps its most widely known achievement. Compared to the original BERT model, it retains 97% of language understanding while being 40% smaller and 60% faster. You can try it [here](https://huggingface.co/distilbert-base-uncased). The same approach has been of applied to other models, such as Facebook's [BART](https://arxiv.org/abs/1910.13461), and you can try DistilBART [here](https://huggingface.co/models?search=distilbart).
	- https://twitter.com/danielgross/status/1619417503561818112?s=46&t=swb0mG9U2bNJt3tlgnWOyw running on apple m2 pro at 50% of an a100 lol
-  [Big Science](https://bigscience.huggingface.co/) project are also very impressive. As visible in this graph included in the [research paper](https://arxiv.org/abs/2110.08207), their T0 model outperforms GPT-3 on many tasks while being 16x smaller.

## directions


CRAMMING: TRAINING A LANGUAGE MODEL ON A SINGLE GPU IN ONE DAY https://arxiv.org/pdf/2212.14034v1.pdf


light BERT model?
https://twitter.com/deepwhitman/status/1610354790772637697?s=46&t=m_H3h2AbaafHjQJEu1cHJw


prompt tuning? https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html?m=1
- https://aclanthology.org/2021.emnlp-main.243.pdf

https://arxiv.org/abs/2301.00774
Massive Language Models Can Be Accurately Pruned in One-Shot

https://arxiv.org/abs/2202.08906 [**ST-MoE: Designing Stable and Transferable Sparse Expert Models**](https://arxiv.org/abs/2202.08906)

Best sparse model paper. Meticulous experiments, comprehensive evals and the goto reference for all sparse modeling


H3 state space models https://twitter.com/mathemagic1an/status/1617620133182332928?s=46&t=I3QXnpdfF5CXpszIYkv4pQ



We present FLAME, a T5-based model trained on Excel formulas that leverages domain insights to achieve competitive performance with a substantially smaller model (60M parameters) and two orders of magnitude less training data. We curate a training dataset using sketch deduplication, introduce an Excel-specific formula tokenizer for our model, and use domain-specific versions of masked span prediction and noisy auto-encoding as pretraining objectives. We evaluate FLAME on formula repair, formula auto-completion, and a novel task called syntax reconstruction. FLAME (60M) can outperform much larger models, such as Codex-Davinci (175B), Codex-Cushman (12B), and CodeT5 (220M), in 6 out of 10 settings.
https://news.ycombinator.com/item?id=34607738


### Finetune Babbage

Our solution is simple, generate a moderately sized corpus of completions made by davinci for a given task, and fine-tune a model like babbage to do the same task. If done correctly you can get near-identical completions (or at least 90% similarity) at a **40x** lower price and around 4-5x better latency. https://www.buildt.ai/blog/incorrectusage
	- if you’re just trying to standardise the format of a prosaic output then you can get away with a couple hundred examples, 
	- if you’re doing logical reasoning, then you’re going to need at least 1000, 
	- if you’re doing DSL work then multiple thousands.


## small hardware

- ESP32 https://news.ycombinator.com/item?id=34632571 https://maxlab.io/store/edge-ai-camera/
- tensorflow lite https://blog.tensorflow.org/2023/02/tensorflow-lite-micro-with-ml-acceleration.html
- jetson nano

## companies in the space

- https://neuton.ai/  Make Edge Devices Intelligent - Automatically build extremely tiny models without coding and embed them into any microcontroller.