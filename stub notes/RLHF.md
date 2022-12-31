
- https://huggingface.co/blog/rlhf
	- progression
		- Basic: Cross Entropy
		- Intermediate: To compensate for the shortcomings of the loss itself people define metrics that are designed to better capture human preferences such as [BLEU](https://en.wikipedia.org/wiki/BLEU) or [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)).
		- HF: While being better suited than the loss function itself at measuring performance these metrics simply compare generated text to references with simple rules and are thus also limited. Wouldn't it be great if we use human feedback for generated text as a measure of performance or go even one step further and use that feedback as a loss to optimize the model?
	- The first [code](https://github.com/openai/lm-human-preferences) released to perform RLHF on LMs was from OpenAI in TensorFlow in 2019.
	- Today, there are already a few active repositories for RLHF in PyTorch that grew out of this. The primary repositories are Transformers Reinforcement Learning ([TRL](https://github.com/lvwerra/trl)), [TRLX](https://github.com/CarperAI/trlx) which originated as a fork of TRL, and Reinforcement Learning for Language models ([RL4LMs](https://github.com/allenai/RL4LMs)).
- https://twitter.com/carperai/status/1582891780931874819
- https://carper.ai/instruct-gpt-announcement/
- WebGPT paper dataset
	- https://huggingface.co/datasets/openai/webgpt_comparisons
	- This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total.


## Quotes

- Human Instrumentality Project https://twitter.com/goodside/status/1603356265391890432