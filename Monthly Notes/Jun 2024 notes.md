
## openai

- minor
	- [criticgpt](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/) writes critiques of ChatGPT responses to help human trainers spot mistakes during RLHF
- nontechnical
	- apple doesnt pay oepnai for chatgpt deal
	- revneue doubels to 3.4b in the last 6 months
	- [mira murati "creepy spyware" reaction](https://www.youtube.com/watch?app=desktop&v=BD0Us5Bn6Lw)

## frontier models

- anthropic
	- [claude 3.5 sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) becomes world top model
	- artifacts [system prompt](https://x.com/elder_plinius/status/1804052791259717665)
	- [claude projects](https://www.anthropic.com/news/projects)
	- [steering api on pilot](https://x.com/alexalbert__/status/1801668464920379648)
		- related: [abliteration](https://huggingface.co/blog/mlabonne/abliteration)
			- Modern LLMs are fine-tuned for safety and instruction-following, meaning they are trained to refuse harmful requests. In their [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), Arditi et al. have shown that this refusal behavior is mediated by a specific direction in the model's residual stream. If we prevent the model from representing this direction, it **loses its ability to refuse requests**. Conversely, adding this direction artificially can cause the model to refuse even harmless requests.
	- [Scaling Monosemanticity podcast discussion](https://www.anthropic.com/research/engineering-challenges-interpretability): 
		- Engineering Problem 1: Distributed Shuffle
			- By the time we were working on Towards Monosemanticity, we had 100TB of training data (100 billion data points, each being 1KB) and shuffling had become a major headache.
		- Engineering Problem 2: Feature Visualization Pipeline
			- For each feature, we want to find a variety of dataset examples that activate it to different levels, exploring its full distribution. Doing this efficiently for millions of features is an interesting distributed systems problem. Originally, all of this ran in a single job – but we quickly scaled beyond that. Below is a sketch of our current approach.
	- [nontechnical](https://x.com/alexalbert__/status/1805617413578539322) 
		- claude sidebar, projects, custom instructions
		


## OS models

- Chrome adds [Window.ai](https://x.com/rauchg/status/1806385778064564622?s=12&t=90xQ8sGy63D2OtiaoGJuww) api with gemini nano
- Apple Intelligence launch

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
- [finetune GPT2 for spam classification](https://x.com/virattt/status/1806377615189528884?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Thread.dev AI-powered Jupyter Notebook](https://github.com/squaredtechnologies/thread) — use local AI to generate and edit code cells, automatically fix errors, and chat with your data

## other launches

- etched launch: https://www.etched.com/announcing-etched
	- Sohu is the world’s first transformer ASIC. One 8xSohu server replaces 160 H100 GPUs.
	- By specializing, Sohu gets unprecedented performance. One 8xSohu server can serve over 500,000 Llama 70B tokens per second.
- luma ai dream machine https://news.ycombinator.com/item?id=40670096
	- [adding keyframes - example prompts](https://x.com/lumalabsai/status/1806435502096310656?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- arc-agi benchmark, [arc prize](https://news.ycombinator.com/item?id=40648960)
	- got 71% or 50% solution https://x.com/bshlgrs/status/1802766374961553887
- Figma AI [(config talk video](https://www.youtube.com/watch?v=_JMmdM00048), [blogpost](https://www.figma.com/blog/introducing-figma-ai/))
	- By clicking **Make Prototype**, you can rapidly turn static mocks into interactive prototypes, making it simpler to bring ideas to life and get stakeholder buy-in. Preview prototypes directly on the canvas to streamline iteration and perfect your designs more efficiently.
	- **Rename Layers** is a seemingly small feature that can save designers hours of monotonous work over the course of a project—helping keep your files organized and developer-ready.
	- We often talk about the [blank canvas problem](https://www.figma.com/blog/introducing-ai-to-figjam/), when you’re faced with a new Figma file and don’t know where to start. **Make Designs** in the Actions panel will generate UI layouts and component options from your text prompts. Just describe what you need, and the feature will provide you with a first draft. By helping you get ideas down quickly, this feature enables you to explore various design directions and arrive at a solution faster.
	- [pitch deck for diagram/figma ai](https://x.com/jsngr/status/1806062691716636983)
	- [“Make Design” AI pulled after criticism](https://techcrunch.com/2024/07/06/figma-pauses-its-new-ai-feature-after-apple-controversy/)
		- The feature, unveiled at the company’s annual Config conference, aimed to jumpstart the design process by generating UI layouts and components from text prompts but faced criticism after it seemingly mimicked the layout of Apple’s Weather app.
- [hugginface open llm leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/blog)
	- gsm8k, truthfulqa are contaminated in instruction datasets
	- models saturated hellaswag, mmlu, arc
	- mmlu has errors
	- new: MMLU-Pro, GPQA, MuSR, MATH, IFEval, BBH
	- Reporting a fairer average for ranking: using normalized scores
	-  some models appear to have a relatively stable ranking (in bold below): Qwen-2-72B instruct, Meta’s Llama3-70B instruct, 01-ai’s Yi-1.5-34B chat, Cohere’s Command R + model, and lastly Smaug-72B, from AbacusAI.
- [midjourney launches personalization](https://x.com/nickfloats/status/1800718391961170356?utm_source=thesephist&utm_medium=email&utm_campaign=maps-and-compasses)
- [fixie ai ultravox](https://x.com/juberti/status/1798898986289684849) - *open source* speech to speech model — understands non-textual speech elements — paralinguistic information. it can pick up on tone, pauses, and more! 

## fundraising

- [cohere $450m at $5b valuation](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)
- [pika $80m series B](https://www.washingtonpost.com/technology/2024/06/04/pika-funding-openai-sora-google-video/) ([bloomberg](https://www.bloomberg.com/news/articles/2024-06-05/spark-capital-jared-leto-back-ai-video-startup-pika))
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
- [curse of the capacity gap](https://x.com/giffmana/status/1806598836959215794)
- [the end of software essay by chris paik](https://x.com/cpaik/status/1796633683908005988)

good reads

- [A Picture is Worth 170 Tokens: How Does GPT-4o Encode Images?](https://www.oranlooney.com/post/gpt-cnn/)
- [Noam Shazeer post on optimizing inference at C.ai](https://research.character.ai/optimizing-inference/?ref=blog.character.ai)
	- [local + global attention being a pattern with gemma 2 and noam](https://x.com/_xjdr/status/1806395387584098633?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Jason Wei, HWChung Stanford lecture](https://x.com/_jasonwei/status/1800967397647503567?s=12&t=90xQ8sGy63D2OtiaoGJuww): simply and clearly explain why language models work so well, purely via intuitions.
- sigma-gpt https://news.ycombinator.com/item?id=40608413
	- The authors randomly permute (i.e., shuffle) input tokens in training and add two positional encodings to each token: one with the token's position and another with the position of the token to be predicted. Otherwise, the model is a standard autoregressive GPT. The consequences of this seemingly "simple" modification are significant:
	- The authors can prompt the trained model with part of a sequence and then decode the missing tokens, all at once, in parallel, regardless of order -- i.e., the model can in-fill in parallel.
	- The authors can compute conditional probability densities for every missing token in a sequence, again in parallel, i.e., densities for all missing tokens at once.
	- The authors propose a rejection-sampling method for generating in-fill tokens, again in parallel. Their method seems to work well in practice.
- 