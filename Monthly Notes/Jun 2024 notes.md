
## openai

- nontechnical
	- apple doesnt pay oepnai for chatgpt deal
	- revneue doubels to 3.4b in the last 6 months

## frontier models

- claude 3.5 sonnet 
	- artifacts [system prompt](https://x.com/elder_plinius/status/1804052791259717665)
	- [nontechnical](https://x.com/alexalbert__/status/1805617413578539322) 
		- claude sidebar, projects, custom instructions

## notable reads

- [A Picture is Worth 170 Tokens: How Does GPT-4o Encode Images?](https://www.oranlooney.com/post/gpt-cnn/)
- sigma-gpt https://news.ycombinator.com/item?id=40608413
	- The authors randomly permute (i.e., shuffle) input tokens in training and add two positional encodings to each token: one with the token's position and another with the position of the token to be predicted. Otherwise, the model is a standard autoregressive GPT. The consequences of this seemingly "simple" modification are significant:
	- The authors can prompt the trained model with part of a sequence and then decode the missing tokens, all at once, in parallel, regardless of order -- i.e., the model can in-fill in parallel.
	- The authors can compute conditional probability densities for every missing token in a sequence, again in parallel, i.e., densities for all missing tokens at once.
	- The authors propose a rejection-sampling method for generating in-fill tokens, again in parallel. Their method seems to work well in practice.

## open models

- [qwen2 release](https://qwenlm.github.io/blog/qwen2/)
	- Pretrained and instruction-tuned models of 5 sizes, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and Qwen2-72B;
	- Having been trained on data in 27 additional languages besides English and Chinese;
	- Extended context length support up to 128K tokens with Qwen2-7B-Instruct and Qwen2-72B-Instruct.
	- 5️⃣ Sizes: 0.5B, 1.5B, 7B, 57B-14B (MoE), 72B as Base & Instruct versions
	- 🪟 Context: 32k for 0.5B & 1.5B, 64k for 57B MoE, 128k for 7B and 72B
	- 🌎 Multilingual in 29 Languages, including European, Middle East, and Asian.
	  📜 Released under Apache 2.0 except 72B version (still commercially useable)
	  🏆 72B: MMLU 82.3; IFEval 77.6; MT-Bench 9.12; 86.0 HumanEval
	  🥇7B: MMLU 70.5; MT-Bench 8.41; HumanEval 79.9
- [Mamba-2 release](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
	- https://arxiv.org/abs/2405.21060
	- https://x.com/_albertgu/status/1797651223035904355
	- https://x.com/tri_dao/status/1797650443218436165
- [stable diffusion 3 medium](https://stability.ai/news/stable-diffusion-3-medium)


## open tooling

- [plandex](https://github.com/plandex-ai/plandex) - a reliable and developer-friendly AI coding agent in your terminal. It can plan out and complete large tasks that span many files and steps.
- [R2R - rag2riches](https://github.com/SciPhi-AI/R2R) -  a prod-ready RAG (Retrieval-Augmented Generation) engine with a RESTful API. R2R includes hybrid search, knowledge graphs, and more.

## other launches

- etched launch: https://www.etched.com/announcing-etched
	- Sohu is the world’s first transformer ASIC. One 8xSohu server replaces 160 H100 GPUs.
	- By specializing, Sohu gets unprecedented performance. One 8xSohu server can serve over 500,000 Llama 70B tokens per second.
- luma ai dream machine https://news.ycombinator.com/item?id=40670096
- arc-agi benchmark, [arc prize](https://news.ycombinator.com/item?id=40648960)
	- got 71% or 50% solution https://x.com/bshlgrs/status/1802766374961553887
- [hugginface open llm leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/blog)
	- gsm8k, truthfulqa are contaminated in instruction datasets
	- models saturated hellaswag, mmlu, arc
	- mmlu has errors
	- new: MMLU-Pro, GPQA, MuSR, MATH, IFEval, BBH
	- Reporting a fairer average for ranking: using normalized scores
	-  some models appear to have a relatively stable ranking (in bold below): Qwen-2-72B instruct, Meta’s Llama3-70B instruct, 01-ai’s Yi-1.5-34B chat, Cohere’s Command R + model, and lastly Smaug-72B, from AbacusAI.
- [midjourney launches personalization](https://x.com/nickfloats/status/1800718391961170356?utm_source=thesephist&utm_medium=email&utm_campaign=maps-and-compasses)

## fundraising

- [cohere $450m at $5b valuation](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)
- [ESM3](https://x.com/soumithchintala/status/1805641549499212259) - EvolutionaryScale: 
ESM3 is a generative language model for programming biology. In experiments, we found ESM3 can simulate 500M years of evolution to generate new fluorescent proteins.

## discussions and good reads


- [leopold aschenbrenner's Trillion Dollar Cluster essay](https://situational-awareness.ai/)
- [cost of self hosting Llama 3](https://blog.lytix.co/posts/self-hosting-llama-3)
	- Assuming 100% utilization of your model Llama-3 8B-Instruct model costs about $17 dollars per 1M tokens when self hosting with EKS, vs ChatGPT with the same workload can offer $1 per 1M tokens. 
	- Choosing to self host the hardware can make the cost <$0.01 per 1M token that takes ~5.5 years to break even.
- [A Picture is Worth 170 Tokens: How Does GPT-4o Encode Images?](https://www.oranlooney.com/post/gpt-cnn/)
	- Here’s a [fact](https://openai.com/api/pricing/): GPT-4o charges 170 tokens to process each `512x512` tile used in high-res mode. At ~0.75 tokens/word, this suggests a picture is worth about 227 words—only a factor of four off from the traditional saying.
- forcing AI on to us
	- msft recall default https://news.ycombinator.com/item?id=40610435
		- and delay https://news.ycombinator.com/item?id=40683210
- apple intelligence
	- [ipad calculator is nutes](https://x.com/levie/status/1800224021193396594)
	- [talaria tool](https://buttondown.email/ainews/archive/ainews-talaria-apples-new-mlops-superweapon-4066/)
- perplexity - forbes attribution issue
- [books4 dataset](https://web.archive.org/web/20240519104217/https://old.reddit.com/r/datasets/comments/1cvi151/ai_books4_dataset_for_training_llms_further/)
- [together ai mixture of agents](https://www.together.ai/blog/together-moa)
	- Our reference implementation, Together MoA, significantly surpass GPT-4o 57.5% on AlpacaEval 2.0 with a score of 65.1% using only open source models. While Together MoA achieves higher accuracy, it does come at the cost of a slower time to first token; reducing this latency is an exciting future direction for this research.
