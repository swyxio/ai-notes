## opinions

large models are demoralizing https://www.reddit.com/r/MachineLearning/comments/wiqjxv/d_the_current_and_future_state_of_aiml_is/

According to Alex Graveley, one of the creators of Github’s Copilot, there is a 1% drop in completions for every additional 10ms of latency. https://www.buildt.ai/blog/incorrectusage

naveen rao https://twitter.com/NaveenGRao/status/1625176665746964480 Smaller models w/ some clever precomputing of results is needed.


## history

- Computer Vision practitioners will remember when [SqueezeNet](https://arxiv.org/abs/1602.07360) came out in 2017, achieving a 50x reduction in model size compared to [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), while meeting or exceeding its accuracy.
- [DistilBERT](https://arxiv.org/abs/1910.01108) is perhaps its most widely known achievement. Compared to the original BERT model, it retains 97% of language understanding while being 40% smaller and 60% faster. You can try it [here](https://huggingface.co/distilbert-base-uncased). The same approach has been of applied to other models, such as Facebook's [BART](https://arxiv.org/abs/1910.13461), and you can try DistilBART [here](https://huggingface.co/models?search=distilbart).
	- https://twitter.com/danielgross/status/1619417503561818112?s=46&t=swb0mG9U2bNJt3tlgnWOyw running on apple m2 pro at 50% of an a100 lol
-  [Big Science](https://bigscience.huggingface.co/) project are also very impressive. As visible in this graph included in the [research paper](https://arxiv.org/abs/2110.08207), their T0 model outperforms GPT-3 on many tasks while being 16x smaller.
- [MiniLM](https://twitter.com/abacaj/status/1633127399930974208?s=46&t=90xQ8sGy63D2OtiaoGJuww) 100m BERT down to 22m with 80% of performance 
- https://minimaxir.com/2023/03/new-chatgpt-overlord/
	- A few years ago, I released [aitextgen](https://github.com/minimaxir/aitextgen), a Python package designed to allow people to train their own custom small AI on their own data for unique use cases. However, soon after, it turned out that GPT-3 with the right prompt could do much better at bespoke generation than a custom model in addition to allowing out-of-domain inputs, even moreso with text-davinci-003. Now with the ChatGPT API making the cost similar to hosting a small model, it’s harder for me to be motivated to continue maintaining the package without first finding another niche.
- https://www.nature.com/articles/d41586-023-00641-w
	- smaller models trained on more data do better than bigger models trained on fewer data[5](https://www.nature.com/articles/d41586-023-00641-w#ref-CR5) (see ‘Different routes to scale’). For example, DeepMind’s Chinchilla model has 70 billion parameters, and was trained on 1.4 trillion tokens, whereas its 280-billion-parameter Gopher model, was trained on 300 billion tokens. Chinchilla outperforms Gopher on tasks designed to evaluate what the LLM has learnt.
	- HOWEVER 
		- The ability to respond to chain-of-thought prompts shows up only in LLMs with more than about 100 billion parameters.
		- cant follow flips https://twitter.com/JerryWeiAI/status/1633548780619571200?s=20

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


snorkel https://snorkel.ai/better-not-bigger-how-to-get-gpt-3-quality-at-0-1-the-cost/
	- We took on a complex 100-way legal classification benchmark task, and with Snorkel Flow and Data-Centric Foundation Model Development, we achieved the same quality as a fine-tuned GPT-3 model with a deployment model that: 
		-   Is 1,400x smaller.
		-   Requires <1% as many ground truth (GT) labels. 
		-   Costs 0.1% as much to run in production.
	- 


H3 state space models https://twitter.com/mathemagic1an/status/1617620133182332928?s=46&t=I3QXnpdfF5CXpszIYkv4pQ
Hyena models https://twitter.com/MichaelPoli6/status/1633167040130453505


-   Parameter-Efficient Fine-tuning (PEFT). The number of params needed to **fine-tune Flan-T5-XXL** is now 9.4M. About 7X fewer than AlexNet. ([link](https://flight.beehiiv.net/v2/clicks/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJodHRwczovL2h1Z2dpbmdmYWNlLmNvL2Jsb2cvcGVmdCIsInBvc3RfaWQiOiJmYjU1ZTM2OC1hZjIyLTRlOWEtOTA1MS1iMTcwY2ZhYjBkMGQiLCJwdWJsaWNhdGlvbl9pZCI6IjQ0N2Y2ZTYwLWUzNmEtNDY0Mi1iNmY4LTQ2YmViMTkwNDVlYyIsInZpc2l0X3Rva2VuIjoiZDQxYjQ5NzktMTMwNy00NWViLTkyZTQtNTgxODg1YmNlMzZiIiwiaWF0IjoxNjc2NDQ2NjQ4LjU3NiwiaXNzIjoib3JjaGlkIn0.q5Tuik3xgVAB4Ymd983PN1MOZX3ni5KdiHkD-TcMtmk))

We present FLAME, a T5-based model trained on Excel formulas that leverages domain insights to achieve competitive performance with a substantially smaller model (60M parameters) and two orders of magnitude less training data. We curate a training dataset using sketch deduplication, introduce an Excel-specific formula tokenizer for our model, and use domain-specific versions of masked span prediction and noisy auto-encoding as pretraining objectives. We evaluate FLAME on formula repair, formula auto-completion, and a novel task called syntax reconstruction. FLAME (60M) can outperform much larger models, such as Codex-Davinci (175B), Codex-Cushman (12B), and CodeT5 (220M), in 6 out of 10 settings.
https://news.ycombinator.com/item?id=34607738


### Finetune Babbage

Our solution is simple, generate a moderately sized corpus of completions made by davinci for a given task, and fine-tune a model like babbage to do the same task. If done correctly you can get near-identical completions (or at least 90% similarity) at a **40x** lower price and around 4-5x better latency. https://www.buildt.ai/blog/incorrectusage
	- if you’re just trying to standardise the format of a prosaic output then you can get away with a couple hundred examples, 
	- if you’re doing logical reasoning, then you’re going to need at least 1000, 
	- if you’re doing DSL work then multiple thousands.

## LoRA

Dylora https://news.ycombinator.com/item?id=35514228


## running in browser

- https://xenova.github.io/transformers.js/ 

## edge/iot/small hardware

- ESP32 https://news.ycombinator.com/item?id=34632571 https://maxlab.io/store/edge-ai-camera/
- tensorflow lite https://blog.tensorflow.org/2023/02/tensorflow-lite-micro-with-ml-acceleration.html
- jetson nano

https://overcast.fm/+vScMSR_PE/33:07
- bandwidth, latency, economics, reliability, privacy
- tensorflow lite components https://overcast.fm/+vScMSR_PE/45:29

## companies in the space

- https://neuton.ai/  Make Edge Devices Intelligent - Automatically build extremely tiny models without coding and embed them into any microcontroller.
- Edge Impulse - Daniel Situy