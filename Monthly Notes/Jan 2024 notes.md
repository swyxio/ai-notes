## Synthetic data

- Synthetic data ['is the future'](https://huggingface.co/datasets/andersonbcdefg/synthetic_retrieval_tasks) from Nous Research founder
- Microsoft trained a text embedding model from Mistral-7B that topped the MTEB  leaderboard [using synthetic data](https://twitter.com/andersonbcdefg/status/1742613575217156547)


## openai

- gpt store launch
	- [pending](https://news.ycombinator.com/item?id=38870249) , [launched on Jan 10](https://twitter.com/sama/status/1745135061731803571)
	- new docs for [Actions for GPTS](https://platform.openai.com/docs/actions/introduction)
- launched and rolled back [GPT personalization and Temporary Chat](https://x.com/AndrewCurran_/status/1744923452572852608?s=20)
- [GPT builder is itself a GPT](https://help.openai.com/en/articles/8770868-gpt-builder)
- [API key based usage tracking](https://twitter.com/OfficialLoganK/status/1743401083920097432) - just tracking, no limits yet
- [removed usage policy against "military and warfare" use](https://theintercept.com/2024/01/12/open-ai-military-ban-chatgpt/)
- announced new [preparedness/safety framework](https://openai.com/safety/preparedness)


## fundraising

- [perplexity series b](https://twitter.com/perplexity_ai/status/1742915781690798290) - 74m at 520m valuation
	- 10m MAU, 500m queries in 2023
	- https://blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round
	- https://www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger-using-ai-to-try-to-upend-internet-search-0859bda6?mod=hp_lead_pos4
- [Quora raised $75m from a16z](https://x.com/adamdangelo/status/1744805602436825334?s=20): "his funding will be used to accelerate the growth of Poe, and we expect the majority of it to be used to pay bot creators through our recently-launched creator monetization program."


## open source tooling

- langchain
	- [v0.1](https://twitter.com/LangChainAI/status/1744411643482951829) launched
	- graph agents
- Llamaindex
	- [Semantic Chunking Llamapack](https://twitter.com/jerryjliu0/status/1745486856291266821?s=12&t=90xQ8sGy63D2OtiaoGJuww) (idea from Greg Kamradt) How it works:
		-   Split text into sentences.
		-   For each sentence, generate an embedding.
		-   Measure cosine distance between each pair of consecutive sentences.
		-   Get the 95% percentile cosine distance, set that as the threshold.
		-   Create a new chunk if the cosine distance of a sentence compared to prev. exceeds that threshold.
- open interpreter 0.2 - vision models, and an api
	- https://twitter.com/hellokillian/status/1743418943120040109
	- https://api.openinterpreter.com/ The Open Interpreter Project has developed (and here freely hosts) an API which is capable of locating visual controls with single-pixel precision.


## models

- [DeepSeekMOE](https://x.com/deepseek_ai/status/1745304852211839163?s=46&t=90xQ8sGy63D2OtiaoGJuww) 2.7b model scaled up to 16B using 2T tokens. 145b MOE on the way  
	- Compared w Mixtral
		- 64 experts vs 8
		- 8 activated experts vs 2 
		- 1 "shared" expert is always activated 
		- 0.24b params per expert vs ~7b 
		- 1.9b total activated params per forward pass vs ~14b
	- beats all other OSS models at 2.7B active param size
	- [notes from Omar Sanseviero](https://twitter.com/osanseviero/status/1745402823682970036)
		- Certain Experts are redundant - they have common knowledge. - So they isolate experts that work as shared experts - they are always activated and reduce redundancy among routed experts - Helps with parameter efficiency
- Phixtral - [MoE of 2 to 4 finetuned models](https://twitter.com/maximelabonne/status/1744867841436700850) - made from dolphin-2_6-phi-2, phi-2-dpo, phi-2-sft-dpo-gpt4_en-ep1, phi-2-coder
- Bagel-34B - new mixtral/merged finetunes from Jon Durbin
	- [uses a bunch of synthetic data](https://github.com/jondurbin/bagel)
	- [comparison from /r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1916896/llm_comparisontest_confirm_leaderboard_big_news/) vs ~~**Mixtral**~~ Yi MoE
- [Mixtral-medium](https://twitter.com/lmsysorg/status/1745061423724875891?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1745061423724875891%7Ctwgr%5E58a43f98e08b74e94594e238390ee283b99e9430%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fspacesdashboard.com%2Fspace%2F1YpKkwDbXPrKj%2Fvirtual-grass-touching-not-recorded) has now beat Claude and is second only to GPT4 on LMSys
- Surya - [mutlilingual text line detection model](https://x.com/vikparuchuri/status/1745876562371903790?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- Text detection is step 1 in building a GPU-accelerated OCR model that is more accurate than tesseract.  Step 2 is to build the text recognition system - I'll be working on that in the next couple of weeks.


## other launches

- [Rabbit R-1 $200 LLM smartphone launched at CES](https://news.ycombinator.com/item?id=38930126)
	- highly produced 30min video, product seems polished, but people mostly unconvinced this needs to be a separate device from phone.
	- [open source clone effort with openinterpreter](https://twitter.com/hellokillian/status/1745875973583896950)
- [Together Embeddings](https://x.com/togethercompute/status/1745500191103553794?s=20)
	- just serving leading oss models incl M2-BERT 32k, UAE-large-v1 and BGE-Base-En-v1.5
	- 3 new embeddings models using Monarch Mixer architecture, enabling long context embeddings up to 32k!
	- <4x lower pricing than OpenAI/Cohere
	- integrations with MongoDB Atlas, Langchain, Llamaindex

## misc reads 

- interesting/interpretability
	- [Chess-GPT's Internal World Model](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html)
		- A 50 million parameter GPT trained on 5 million games of chess learns to play at ~1300 ELO in one day on 4 RTX 3090 GPUs. This model is only trained to predict the next character in PGN strings (1.e4 e5 2.Nf3 …) and is never explicitly given the state of the board or the rules of chess. Despite this, in order to better predict the next character, it learns to compute the state of the board at any point of the game, and learns a diverse set of rules, including check, checkmate, castling, en passant, promotion, pinned pieces, etc. In addition, to better predict the next character it also learns to estimate latent variables such as the ELO rating of the players in the game.
		- [Tweet thread](https://twitter.com/a_karvonen/status/1743666230127411389): "I trained Chess-GPT, a 50M parameter LLM, to play at 1500 ELO. We can visualize its internal state of the board. In addition, to better predict the next character it estimates the ELO of the players involved"
- NIST paper: [Adversarial Machine Learning - A Taxonomy and Terminology of Attacks and Mitigations](https://twitter.com/rez0__/status/1743266573668757568)
	- link to [paper](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf)  
- Discussions:
	- Simonw vs Karpathy - [AI vs IA](https://x.com/karpathy/status/1744062845426532473?s=20)
		- followup https://twitter.com/karpathy/status/1744179910347039080
	- DPO paper got [Standing ovation from Andrew Ng](https://twitter.com/andrewyng/status/1745516258697863259?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- https://www.deeplearning.ai/the-batch/issue-231/
		- Via clever mathematical insight, the authors show that given an LLM, there is a specific reward function for which that LLM is optimal. DPO then trains the LLM directly to make the reward function (that’s now implicitly defined by the LLM) consistent with the human rankings. So you no longer need to deal with a separately represented reward function, and you can train the LLM directly to optimize the same objective as RLHF.
	- [story of the acquisition of Gradio by Huggingface](https://twitter.com/abidlabs/status/1745533306492588303?s=12&t=90xQ8sGy63D2OtiaoGJuww)
