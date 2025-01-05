## Synthetic data

- Synthetic data ['is the future'](https://huggingface.co/datasets/andersonbcdefg/synthetic_retrieval_tasks) from Nous Research founder
- Microsoft trained a text embedding model from Mistral-7B that topped the MTEB  leaderboard [using synthetic data](https://twitter.com/andersonbcdefg/status/1742613575217156547)

ways we are juicing models
- finetuning and merging  https://github.com/hiyouga/LLaMA-Factory
	- [Deepmind WARM - Weight Averaged Reward Models](https://x.com/ramealexandre/status/1749719476310843794?s=20)
	- [multiple DPO RMs merged](https://x.com/ramealexandre/status/1749719489275392372?s=20)
- superchinchilla
	- https://arxiv.org/pdf/2401.00448.pdf The tl;dr is you usually can save [20% flops lifetime cost by training a 30% smaller model for 2-3x as long](https://x.com/andrew_n_carr/status/1742017610500444478?s=20).
	- https://news.ycombinator.com/item?id=37383413

tech layoffs - [google](https://news.ycombinator.com/item?id=38948444) et al


## openai

- gpt store launch
	- [pending](https://news.ycombinator.com/item?id=38870249) , [launched on Jan 10](https://twitter.com/sama/status/1745135061731803571), [see replies for what people are working on](https://twitter.com/officiallogank/status/1744396647432323079?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- new docs for [Actions for GPTS](https://platform.openai.com/docs/actions/introduction)
	- [featured GPTS](https://x.com/chatgptapp/status/1750316948444086697)
- launched and rolled back [GPT personalization and Temporary Chat](https://x.com/AndrewCurran_/status/1744923452572852608?s=20)
- you can now [@mention different GPTs in a single convo](https://twitter.com/officiallogank/status/1752392820181107115?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [GPT builder is itself a GPT](https://help.openai.com/en/articles/8770868-gpt-builder)
- [API key based usage tracking](https://twitter.com/OfficialLoganK/status/1743401083920097432) - just tracking, no limits yet
- [removed usage policy against "military and warfare" use](https://theintercept.com/2024/01/12/open-ai-military-ban-chatgpt/)
- announced new [preparedness/safety framework](https://openai.com/safety/preparedness)
- pushed back on [the NYT lawsuit](https://openai.com/blog/openai-and-journalism)
- sama tsmc https://x.com/basedbeffjezos/status/1748903066735354030?s=46&t=90xQ8sGy63D2OtiaoGJuww
- [first university partnership with Arizona State University](https://www.cnbc.com/2024/01/18/openai-announces-first-partnership-with-a-university.html) - ASU plans to build a personalized AI tutor for students, allow students to create AI avatars for study help and broaden the university’s prompt engineering course.
- [preparedness for 2024 elections](https://openai.com/blog/how-openai-is-approaching-2024-worldwide-elections)
	- usage policies -> [guardian_tool for usage policies](https://dmicz.github.io/machine-learning/chatgpt-election-update/) silently added
	- CCPA digital credentials - encodes provenance for DallE3 images
	- access to voting information in chatgpt
- sama at davos
	- https://www.axios.com/2024/01/17/sam-altman-davos-ai-future-interview
	- ([video](https://www.youtube.com/watch?v=QFXp_TU-bO8))
	- Altman said his top priority right now is launching the new model, likely to be called GPT-5.
- [GPT3/4 Turbo and New embedding models](https://openai.com/blog/new-embedding-models-and-api-updates)
	- gpt-3.5-turbo-0125 - $0.5 per million input, $1.5 per million output (25-50% cheaper than gpt-3.5-turbo-1106)
		- 16k tokens - no longer separate 4k and 16k moels for GPT3.5T
		- This model will also have various improvements including higher accuracy at responding in requested formats and a fix for [a bug](https://community.openai.com/t/gpt-4-1106-preview-is-not-generating-utf-8/482839) which caused a text encoding issue for non-English language function calls.
	- gpt-4-0125-preview - same price as other GPT4T models
		- 128k token context, same as before
		- This model completes tasks like code generation more thoroughly than the previous preview model and is intended to reduce cases of “laziness” where the model doesn’t complete a task. The new model also includes the fix for the bug impacting non-English UTF-8 generations.
		- We plan to launch GPT-4 Turbo with vision in general availability in the coming months.
	- text-embedding-3-small - 1536 dims, on MTEB 1.3 better then ada-002. 20% price of ada-002 
	- text-embedding-3-large - 3072 dims, on MTEB 3.6 better then ada-002. bit more expensive than ada-002
	- developers can now assign permissions to API keys from the [API keys page](https://platform.openai.com/api-keys). For example, a key could be assigned read-only access to power an internal tracking dashboard, or restricted to only access certain endpoints.
	-  the usage dashboard and usage export function now expose metrics on an API key level after [turning on tracking](https://platform.openai.com/api-keys). This makes it simple to view usage on a per feature, team, product, or project level, simply by having separate API keys for each.
- [fun content policy jailbreak](https://twitter.com/jeremyntrimble/status/1745087312344604941?s=12&t=90xQ8sGy63D2OtiaoGJuww) - ChatGPT won’t identify people in images even if they’re famous, but if you put them in a picture with a cartoon it will identify them. Cartoon has to be on the left or it won’t work.


## google/gemini

- gemini pro doing better vs gpt4T - https://twitter.com/lmsysorg/status/1750921228012122526
	- due to online rag https://twitter.com/Weyaxi/status/1751380303988359241
	- and a different finetune https://x.com/asadovsky/status/1750983142041911412?s=20
- but does [VERY poorly on needle in haystack test](https://twitter.com/aparnadhinak/status/1744771295940669689?s=12&t=90xQ8sGy63D2OtiaoGJuww)

## anthropic

- [sleeper agents](https://fxtwitter.com/AnthropicAI/status/1745854907968880970)
- https://transformer-circuits.pub/2024/jan-update/index.html

## fundraising

- [ElevenLabs series B](https://twitter.com/elevenlabsio/status/1749435751656231065) - 80m at 1b valuation with a16z and nat/dan
	- [seed pitch deck](https://x.com/chiefaioffice/status/1749385259517440495?s=20)
- [perplexity series b](https://twitter.com/perplexity_ai/status/1742915781690798290) - 74m at 520m valuation
	- 10m MAU, 500m queries in 2023
	- https://blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round
	- https://www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger-using-ai-to-try-to-upend-internet-search-0859bda6?mod=hp_lead_pos4
- [Quora raised $75m from a16z](https://x.com/adamdangelo/status/1744805602436825334?s=20): "his funding will be used to accelerate the growth of Poe, and we expect the majority of it to be used to pay bot creators through our recently-launched creator monetization program."
- [Luma Labs - $43m Series B from a16z](https://lumalabs.ai/series-b)
	- [Luma Labs Genie 1.0](https://twitter.com/LumaLabsAI/status/1744778363330535860) -  a text-to-3d model capable of creating any 3d object you can dream of in under 10 seconds with materials, quad mesh retopology, variable polycount, and in all standard formats
- Elon denying x.ai 6b at 20b valuation https://www.reuters.com/technology/elon-musks-ai-start-up-seeks-raise-up-6-bln-ft-2024-01-26/
- [Cohere in talks to raise $1b](https://www.ft.com/content/631e91f6-4b24-4d4f-80cc-503be97a79c8)
- Multion "[came out of stealth](https://twitter.com/divgarg9/status/1744771026259718402?s=12&t=90xQ8sGy63D2OtiaoGJuww)" with funding announcements but no specific numbers
- [Tab announced $1.9m seed](https://twitter.com/avischiffmann/status/1745048556891783227?s=12&t=90xQ8sGy63D2OtiaoGJuww) - your super intelligent sidekick. [Forbes](https://www.fastcompany.com/91007630/avi-schiffmanns-tab-ai-necklace-has-raised-1-9-million-to-replace-god)
- [Leap API - $1.4m seed](https://twitter.com/leap_api/status/1744786661094072700?s=12&t=90xQ8sGy63D2OtiaoGJuww) - no-code AI workflow builder (compare with respell etc)


## open source tooling & projects


- langchain
	- [v0.1](https://twitter.com/LangChainAI/status/1744411643482951829) launched:
		- separating out langchain-core and separating out partner packages (either into langchain-community or standalone partner packages) from langchain. 
		- 👀Observability: Building complex LLM applications is hard. In order to best debug, you need to know the exact steps that were taken and the input/output at each step. Through a tight integration with LangSmith, LangChain has best-in-class observability
		- ↔️Integrations: With nearly 700 integrations, no matter what tech stack you want to use, LangChain supports it
		- 🔗Composability: With LangChain Expression Language, it's easy (and fun!) to create arbitrary chains, bringing you all the benefits of a data orchestration framework
		- 🎏Streaming: We've invested a lot in making sure that streaming is supported in a first class way for all chains created with LangChain Expression Language - including streaming of intermediate steps
		- 🧱Output parsing: Getting the LLM to return information in a certain format is key for enabling it to take actions.
		- 🔎Retrieval: adding advanced yet production ready methods for RAG, including text-splitting, retrieval, and an indexing pipeline
		- 🤖Tool Use + Agents: collection of agents (decide what actions to take), collection of tools, easy way to define tools
		- [llamaindex v0.10 did a similar split move in feb](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8?source=collection_home---6------0-----------------------)
	- [langgraph graph agents](https://github.com/langchain-ai/langgraph?ref=blog.langchain.dev)
		- [compare with autogen](https://www.youtube.com/watch?v=v9fkbTxPzs0&t=2s)
- [Crew AI](https://github.com/joaomdmoura/crewAI) - trended all month due to [good demoing](https://twitter.com/joaomdmoura/status/1744031203995316712?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.
	- Builds atop langchain, Harrison quite supportive
- [Griptape](https://github.com/griptape-ai/griptape) - a langchain alternative
- Llamaindex
	- [Semantic Chunking Llamapack](https://twitter.com/jerryjliu0/status/1745486856291266821?s=12&t=90xQ8sGy63D2OtiaoGJuww) (idea from Greg Kamradt) How it works:
		-   Split text into sentences.
		-   For each sentence, generate an embedding.
		-   Measure cosine distance between each pair of consecutive sentences.
		-   Get the 95% percentile cosine distance, set that as the threshold.
		-   Create a new chunk if the cosine distance of a sentence compared to prev. exceeds that threshold.
	- JSONalyze https://docs.llamaindex.ai/en/latest/examples/query_engine/JSONalyze_query_engine.html#
- transformers 
	- [ngram speculative decoding now](https://x.com/abacaj/status/1749612925973680426?s=20) supported. 3x speedup with no extra resources needed
- [Lepton Search demo](https://twitter.com/jiayq/status/1750935085564764224?s=12&t=90xQ8sGy63D2OtiaoGJuww) aka "open source perplexity ai"
	- Over the weekend, we built a demo for conversational search with <500 lines of python, and it's live at [https://search.lepton.run](https://t.co/l7q4GGCQOH).
	- The backend is a really fast Mixtral-8x7b model served on @LeptonAI with throughput as fast as around 200 tokens / second (if you are lucky to not hit "LLM commute traffic"). Search engine is currently Bing search API. Lepton KV as a serverless storage. A few learnings on the road: (1) search quality is really, really important. A good snippet lead to good summarization. (2) a bit of hallucination actually helps to fill in "common knowledge" not covered in the snippets. (3) open source models work really well for summarization.
- open interpreter 0.2 - vision models, and an api
	- https://twitter.com/hellokillian/status/1743418943120040109
	- https://api.openinterpreter.com/ The Open Interpreter Project has developed (and here freely hosts) an API which is capable of locating visual controls with single-pixel precision.
- Codium - [AlphaCodium](https://www.codium.ai/blog/alphacodium-state-of-the-art-code-generation-for-code-contests/) - beating deepmind alphacoder/deepseek/gpt4 with prompt flow  
	- [itamar's intro video](https://twitter.com/itamar_mar/status/1747957348293824676)
	- https://x.com/karpathy/status/1748043513156272416?s=20
		- Prompt engineering (or rather "Flow engineering") intensifies for code generation. Great reading and a reminder of how much alpha there is (pass@5 19% to 44%) in moving from a naive prompt:answer paradigm to a "flow" paradigm, where the answer is constructed iteratively.
	- https://twitter.com/svpino/status/1747971746047627682
	- https://twitter.com/swyx/status/1748084170537291923
	- can be [prototyped in DSPy](https://twitter.com/CShorten30/status/1751656468879708496/photo/1) ([note correction](https://x.com/CShorten30/status/1751785503244849415?s=20))
- SGLang from LMsys
	-  [code](https://github.com/sgl-project/sglang/) and a [tech report](https://arxiv.org/abs/2312.07104)
	- "our next-generation interface and runtime for LLM inference! It greatly improves the execution and programming efficiency of complex LLM programs by co-designing the front-end language and back-end runtime."
	- On the backend, we propose [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/), a novel technique that automatically handles various patterns of KV cache reuse. 
		- Instead of discarding the KV cache after finishing a generation request, our approach retains the KV cache for both prompts and generation results in a radix tree. This data structure enables efficient prefix search, insertion, and eviction. We implement a Least Recently Used (LRU) eviction policy, complemented by a cache-aware scheduling policy, to enhance the cache hit rate.
	- On the frontend, we designed a flexible prompting language for you to control the generation process.
	- SGLang can perform up to 5x faster than existing systems like Guidance and vLLM on common LLM workloads (agent, reasoning, chat, RAG, few-shot benchmark), while also reducing code complexity.
- [RAGatouille](https://github.com/bclavie/RAGatouille) - RAGatouille focuses on making ColBERT simple to use ([why colbert](https://twitter.com/lateinteraction/status/1743040090400858288)? [speed](https://twitter.com/virattt/status/1749166976033861832?s=12&t=90xQ8sGy63D2OtiaoGJuww). another take [from mark tenenholtz](https://twitter.com/marktenenholtz/status/1751406680535883869)? try [visualization](https://twitter.com/jobergum/status/1751640310642360694))
	- The Information Retrieval research field has recently been booming, and models like ColBERT have been shown to [generalise better](https://arxiv.org/abs/2203.10053) [to new or complex domains](https://aclanthology.org/2022.findings-emnlp.78/) [than dense embeddings](https://arxiv.org/abs/2205.02870), are [ridiculously data-efficient](https://arxiv.org/abs/2309.06131) and are even [better suited to efficiently being trained](https://arxiv.org/abs/2312.09508) [on non-English languages with low amount of data](https://arxiv.org/abs/2312.16144)! Unfortunately, most of those new approaches aren't very well known, and are much harder to use than dense embeddings.
	- [llamapack plugin here](https://twitter.com/jerryjliu0/status/1743077679258320925?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Lumos - Local LLM chrome extension powered by Ollama](https://news.ycombinator.com/item?id=39132766)
- [PolyMind](https://github.com/itsme2417/PolyMind) - multimodal, function calling powered LLM webui. It's designed to be used with Mixtral 8x7B + TabbyAPI and offers a wide range of features including:
	-   Internet searching with DuckDuckGo and web scraping capabilities.
	-   Image generation using comfyui.
	-   Image input with sharegpt4v (Over llama.cpp's server), OCR, and Yolo.
	-   Port scanning with nmap.
	-   Wolfram Alpha integration.
	-   A Python interpreter.
	-   RAG with semantic search for PDF and miscellaneous text files.
- [Datatrove - huggingface’s commoncrawl data pipeline tool](https://github.com/huggingface/datatrove)
- [Huggingface Candle](https://x.com/reach_vb/status/1743698310181917096?s=20) -  Rust based xplatform ML framework (compare with llama.cpp, mlc chat, ollama etc)
	- https://nitro.jan.ai/ also launched but to [negative reviews](https://news.ycombinator.com/item?id=38887531)
- [Portkey](https://news.ycombinator.com/item?id=38911677) - open source AI gateway
	- Blazing fast (9.9x faster) with a tiny footprint (~45kb installed)
	- Load balance across multiple models, providers, and keys
	- Fallbacks make sure your app stays resilient
	- Automatic Retries with exponential fallbacks come by default
	- Plug-in middleware as needed
	- Battle tested over 100B tokens and millions of requests


## models

- Embedding models
	- [dont forget the openai turbo and embedding models](https://openai.com/blog/new-embedding-models-and-api-updates)
	- voyage-code-2 dropped last week -- a new embedding model specifically trained & optimized for code-related applications. they claim it's 16.93% better than ada on code datasets, but it remains to be seen how voyage-code performs compared to the new ada-v3 based on MRL [https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/ "https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/") - thanks Gian on discord
- codellama70b ([Tweet](https://x.com/teortaxestex/status/1752125379303875038?s=46&t=90xQ8sGy63D2OtiaoGJuww), [HN](https://news.ycombinator.com/item?id=39178886))
	- Trained on 1T extra Tokens, with a [few B more tokens for variants](https://x.com/_philschmid/status/1752024648697434422?s=20)
	- 🐍 Fine-tuned Python version
	- 🛠 Fine-tuned Instruct version
	- ✅ Commercial use allowed
	- 🪟 16384 context window
	- mlx and quantized https://x.com/ivanfioravanti/status/1752133596502986829
- [DeepSeekMOE](https://x.com/deepseek_ai/status/1745304852211839163?s=46&t=90xQ8sGy63D2OtiaoGJuww) ([paper](https://arxiv.org/pdf/2401.06066.pdf)) 2.7b model scaled up to 16B using 2T tokens. 145b MOE on the way  
	- Compared w Mixtral
		- 64 experts vs 8
		- 8 activated experts vs 2 
		- 1 "shared" expert is always activated 
		- 0.24b params per expert vs ~7b 
		- 1.9b total activated params per forward pass vs ~14b
	- beats all other OSS models at 2.7B active param size
	- [notes from Omar Sanseviero](https://twitter.com/osanseviero/status/1745402823682970036)
		- Certain Experts are redundant - they have common knowledge. - So they isolate experts that work as shared experts - they are always activated and reduce redundancy among routed experts - Helps with parameter efficiency
- [MoE-Mamba](https://arxiv.org/abs/2401.04081) and [BlackMamba](https://twitter.com/_akhaliq/status/1754723073889120555) - MoE experiments with Mamba models
	- there was kind of a quick rush of Mamba MOE papers hot after Mamba came out - [memed](https://twitter.com/osanseviero/status/1750830189028819175?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- RWKV Eagle-7B (on RKWV v5)
	- [cleanly licensed, multilingual, green](https://twitter.com/RWKV_AI/status/1751797147492888651)
	- [yam peleg writeup](https://twitter.com/Yampeleg/status/1751850391480721693)
- Deepseek coder 7B https://x.com/teortaxestex/status/1752177206283964813?s=46&t=90xQ8sGy63D2OtiaoGJuww
- [WizardCoder 33B](https://twitter.com/WizardLM_AI/status/1742906065359167730) - 79.9% on HumanEval, 78.9% on MBPP, bit behind GPT4 level on most things but still SOTA Code LLM
	- it is a [dead heat between DeepSeekCoder and WizardCoder](https://evalplus.github.io/leaderboard.html) on the benchmarks, after GPT4.
- Phixtral - [MoE of 2 to 4 finetuned models](https://twitter.com/maximelabonne/status/1744867841436700850) - made from dolphin-2_6-phi-2, phi-2-dpo, phi-2-sft-dpo-gpt4_en-ep1, phi-2-coder
	- [try on hf space](https://huggingface.co/spaces/mlabonne/phixtral-chat)
- Bagel-34B - new mixtral/merged finetunes from Jon Durbin
	- [uses a bunch of synthetic data](https://github.com/jondurbin/bagel)
	- [comparison from /r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1916896/llm_comparisontest_confirm_leaderboard_big_news/) vs ~~**Mixtral**~~ Yi MoE
- [Mixtral-medium](https://twitter.com/lmsysorg/status/1745061423724875891?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1745061423724875891%7Ctwgr%5E58a43f98e08b74e94594e238390ee283b99e9430%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fspacesdashboard.com%2Fspace%2F1YpKkwDbXPrKj%2Fvirtual-grass-touching-not-**recorded**) has now beat Claude and is second only to GPT4 on LMSys
- Surya - [mutlilingual text line detection model](https://x.com/vikparuchuri/status/1745876562371903790?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- Text detection is step 1 in building a GPU-accelerated OCR model that is more accurate than tesseract.  Step 2 is to build the text recognition system - I'll be working on that in the next couple of weeks.
	- related: [Vision Grid Transformer](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT): a two-stream multi-modal Vision Grid Transformer for document layout analysis, in which Grid Transformer (GiT) is proposed and pre-trained for 2D token-level and segment-level semantic understanding. By fully leveraging multi-modal information and exploiting pre-training techniques to learn better representation, VGT achieves highly competitive scores in the DLA task, and significantly outperforms the previous state-of-the-arts.
- [Hourglass Diffusion Transformers](https://twitter.com/iScienceLuvr/status/1749624496770973816) - 
	- We introduce a new transformer backbone for diffusion models that can directly generate megapixel images without the need for multiple stages like latent diffusion.
	- O(n) complexity of UNet, but parameter-scalability of transformers.
		- "Based on ablation studies this allows them to maintain or slightly improve FID score compared to Transformer-only diffusion models that don't do U-net like structures, but at 1/10th the computation cost. An incredible feat for sure."
	- [HN](https://news.ycombinator.com/item?id=39107620),[ tweet thread of threads](https://twitter.com/iScienceLuvr/status/1749624496770973816)
	- this arch is of course nice for high-resolution synthesis, but there's some other cool stuff worth mentioning..
	- activations are small! so you can enjoy bigger batch sizes. this is due to the 4x patching we do on the ingress to the model, and the effectiveness of neighbourhood attention in joining patches at the seams.
	- the model's inductive biases are pretty different than (for example) a convolutional UNet's. the innermost levels seem to train easily, so images can have good global coherence early in training.
	- there's no convolutions! so you don't need to worry about artifacts stemming from convolution padding, or having canvas edge padding artifacts leak an implicit position bias.
	- we can finally see what high-resolution diffusion outputs look like _without_ latents! personally I think current latent VAEs don't _really_ achieve the high resolutions they claim (otherwise fine details like text would survive a VAE roundtrip faithfully); it's common to see latent diffusion outputs with smudgy skin or blurry fur. what I'd like to see in the future of latent diffusion is to listen to the Emu paper and use more channels, or a less ambitious upsample.
	- it's a transformer! so we can try applying to it everything we know about transformers, like sigma reparameterisation or multimodality. some tricks like masked training will require extra support in [NATTEN]([https://github.com/SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN)), but we're very happy with its featureset and performance so far.
	- but honestly I'm most excited about the efficiency. there's too little work on making pretraining possible at GPU-poor scale. so I was very happy to see HDiT could succeed at small-scale tasks within the resources I had at home (you can get nice oxford flowers samples at 256x256px with half an hour on a 4090). I think with models that are better fits for the problem, perhaps we can get good results with smaller models. and I'd like to see big tech go that direction too!\
- [Stable LM 2 1.6B](https://x.com/_akhaliq/status/1748533176547369391?s=20)
	- The base model is trained on approximately 2 trillion tokens for two epochs, incorporating multilingual data in English, Spanish, German, Italian, French, Portuguese, and Dutch.
	- “Bit hard to interpret this graph, but TinyLlama < Phi2 < StableLM < Mistral. If so, it's impressive that StableLM bet Phi-2.”
	- [in venturebeat](https://venturebeat.com/ai/stability-ai-unveils-smaller-more-efficient-1-6b-language-model-as-part-of-ongoing-innovation/)
- Google [Lumiere](https://lumiere-video.github.io/): A Space-Time Diffusion Model for Realistic Video Generation
	- yannic kilcher did nice breakdown
- [Long-Context Monarch Mixer models](https://twitter.com/realDanFu/status/1745507410662580388) - new releases of M2-BERT up to 32K context length, as well as embedding versions fine-tuned for long-context retrieval.
	- also some good explanation of why MultipleNegativesRankingLoss is very dependent on the batch size. If your batch is too small, you don’t get enough negative samples, and you get a bad embedding geometry.
	- This poses a problem for training long-context retrieval models – when you’re fine-tuning on long documents, you are limited in batch size due to GPU memory limits (more tokens -> more memory).
	- For context, we typically want batches of size 32 or greater with the contrastive loss – but we’re limited to size 1 at sequence length 32K.
- [DeepMind AlphaGeometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/?utm_source=twitter&utm_medium=social):  an AI system that solves Olympiad geometry problems at a level approaching a human gold-medalist. It was trained solely on 100 million synthetic data examples and marks a breakthrough for AI in mathematical reasoning. 
	- In a benchmarking test of 30 Olympiad geometry problems, AlphaGeometry solved 25 within the standard Olympiad time limit. For comparison, the previous state-of-the-art system solved 10 of these geometry problems, and the average human gold medalist solved 25.9 problems.
	- Published in Nature.
	- 51days from the [$10m IMO Gold prize](https://x.com/8teAPi/status/1747708708732510653?s=20)
- [Adept Fuyu-Heavy](https://www.adept.ai/blog/adept-fuyu-heavy) - "Fuyu-Heavy is the world’s third-most-capable multimodal model, behind only GPT4-V and Gemini Ultra, which are 10-20 times bigger."
	- It excels at multimodal reasoning. To us the killer feature is UI understanding, but it also performs well on more traditional multimodal benchmarks. In particular, Fuyu-Heavy scores higher on the MMMU benchmark than even Gemini Pro.
	- On standard text-based benchmarks, it matches or exceeds the performance of models in the same compute class despite having to devote some of its capacity to image modeling.
	- merely announced, not released. [blog post criticized](https://twitter.com/teortaxestex/status/1750353889499459746?t=90xQ8sGy63D2OtiaoGJuww).
- LiteLlama - open source repro of Llama2 - but 460m params trained with 1T tokens - [not as good as larger llama variants.](https://x.com/op7418/status/1744019832654397890?s=20)
- [Moondream1](https://x.com/vikhyatk/status/1749625143155167702?s=20) - 1.6b param vision language model - [some good demos](https://x.com/vikhyatk/status/1749629523715567654?s=20) - based on LLaVA dataset

## datasets & benchmarks

- [OpenHermes datasets finally published](https://twitter.com/Teknium1/status/1752799892215673313)
- [WebSight](https://t.co/CwRKlkAXhH) - a multimodal dataset featuring 823,000 pairs of synthetically generated HTML/CSS codes along with screenshots of the corresponding rendered websites, to fine-tune GPT4-V-like models.
	- used both Mistral-7B-v0.1 from @MistralAI and Deepseek-Coder-33b-Instruct from @deepseek_ai to generate the website ideas and code
	- discussed in Latent Space Huggingface episode
- [TinyNarrations](https://sfcompute.com/blog/tiny-narrations): a synthetic audio dataset based on [TinyStories](https://arxiv.org/abs/2305.07759). 
		- The data consists of 30 thousand hours of story narrations from the original GPT-4 generated instruct dataset, synthesized with [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) ([try on fal ai](https://twitter.com/jfischoff/status/1743328938955587877?s=12&t=90xQ8sGy63D2OtiaoGJuww)) over three days on one of our [H100 nodes](https://sfcompute.com/signup). The audio was generated in sentence chunks and concatenated into files of approximately 30 minutes. Due to the causal conditioning of the model, no two words are identically pronounced, so it should be non-trivial for a model to extract semantics from the data. In total, the dataset is about 15TB in size, with a validation subset of about 1%. We also include pre-tokenized data using Meta’s Hubert and Encodec models, as used in architectures like AudioLM.
- [AllenAI new CommonGen leaderboard/dataset](https://github.com/allenai/CommonGen-Eval)  ([paper](https://arxiv.org/abs/1911.03705))
	-  Given a set of common concepts (e.g., {dog, frisbee, catch, throw}); the task is to generate a coherent sentence describing an everyday scenario using these concepts (e.g., "a man throws a frisbee and his dog catches it").  
	- The CommonGen task is challenging because it inherently requires 1) relational reasoning with background commonsense knowledge, and 2) compositional generalization ability to work on unseen concept combinations. 
	- Our dataset, constructed through a combination of crowdsourced and existing caption corpora, consists of 79k commonsense descriptions over 35k unique concept-sets. Experiments show that there is a large gap between state-of-the-art text generation models (e.g., T5) and human performance
- [Orca Pairs DPO](https://twitter.com/argilla_io/status/1745057571696693689?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- The original dataset just assumes gpt4/3.5-turbo are always the best response. We know from that's not always the case. Moreover, DPO fine-tuning benefits from diversity of preference pairs.
	- We used distilabel, our open-source AI Feedback framework to build a preference dataset with ratings for each pair and natural language critiques.
	- [outperforms other models with 54% less samples](https://x.com/argilla_io/status/1745057582308339792?s=20)
- [AlpacaEval announced a 2.0](https://twitter.com/yanndubs/status/1744908464873513085?s=12&t=90xQ8sGy63D2OtiaoGJuww) Main changes: 
	- we use GPT-4 turbo as baseline
	- we use the annotator's logprobs to extract weighted win-rates
	- we changed the annotators' prompt
	- we use GPT-4 turbo as annotator
- GPT4V microbenchmark - https://www.gptcheckup.com/
	- We test tasks we know GPT-4 with Vision performs well at (i.e. classification) to measure regressions, as well as tasks GPT-4 with Vision struggles with (i.e. odometer OCR) to measure performance improvements and changes.
	- [known mode collapse issue when OCRing gauges](https://x.com/teortaxesTex/status/1749846266442367426?s=20)
- [Reranking speed/quality benchmark](https://twitter.com/virattt/status/1752098121826361453?s=12&t=90xQ8sGy63D2OtiaoGJuww) - Cohere does very well vs ColBERT, GPT4, mistral-medium. [notebook](https://gist.github.com/virattt/bf13f748c6b4763b6c6215c8659c02f6)
- [Nous Research - new bittensor decentralized leaderboard](https://twitter.com/nousresearch/status/1752051008736550917?s=12&t=90xQ8sGy63D2OtiaoGJuww)

## other launches

- [Rabbit R-1 $200 LLM smartphone launched at CES](https://news.ycombinator.com/item?id=38930126)
	- highly produced 30min video, product seems polished, but people mostly unconvinced this needs to be a separate device from phone.
	- [open source clone effort with openinterpreter](https://twitter.com/hellokillian/status/1745875973583896950)
- [Together Embeddings](https://x.com/togethercompute/status/1745500191103553794?s=20)
	- just serving leading oss models incl M2-BERT 32k, UAE-large-v1 and BGE-Base-En-v1.5
	- 3 new embeddings models using Monarch Mixer architecture, enabling long context embeddings up to 32k!
	- <4x lower pricing than OpenAI/Cohere
	- integrations with MongoDB Atlas, Langchain, Llamaindex
	- we discussed this in [the Together AI episode](https://www.latent.space/p/together)
-  [artificialanalysis.ai](https://news.ycombinator.com/item?id=39014985) - Benchmarks and comparison of LLM AI models and API hosting providers - [swyx tweet](https://twitter.com/swyx/status/1747741795281412133)
	- Martian also launched [leaderboard](https://leaderboard.withmartian.com) but not as good 
- Nightshade/Glaze 1.0 ([tweet](https://twitter.com/alexjc/status/1748754290435395739), [podcast](https://twimlai.com/podcast/twimlai/nightshade-data-poisoning-to-fight-generative-ai/))
	- a tool that turns any image into a data sample that is unsuitable for model training. More precisely, Nightshade transforms images into "poison" samples, so that models training on them without consent will see their models learn unpredictable behaviors that deviate from expected norms, e.g. a prompt that asks for an image of a cow flying in space might instead get an image of a handbag floating in space.
	- [circumventing Nightshade is violation of DMCA](https://twitter.com/alexjc/status/1748754290435395739)
- Vx.dev
	- github actions-driven development
	- vx.dev was initially designed as an open-source alternative to Vercel's v0.dev, so we also have a [blog](https://step-saga-examples.pages.dev/v0-dev-reverse-engineer/) about how we reverse-engineered v0.dev.
	- also has an interesting blogpost on [How I Reverse Engineered Vercel's v0.dev Prompt and Code Optimization Logic](https://step-saga-examples.pages.dev/v0-dev-reverse-engineer/)
- [Rosebud.ai](https://news.ycombinator.com/item?id=38868185) - turn game descriptions into browser games
	- [8 minute demo](https://www.youtube.com/watch?v=h99H3FefxU0)
- Krea.ai launching Portrait, Concept CGI and Cartoon modes
	- [Cerebral Valley interview](https://cerebralvalley.beehiiv.com/p/krea-building-next-frontier-human-creativity)
- Arc Search browses the web for you, and then builds you the webpage you wanted.[https://www.theverge.com/2024/1/28/24053882/arc-search-browser-web-app-ios](https://www.theverge.com/2024/1/28/24053882/arc-search-browser-web-app-ios "https://www.theverge.com/2024/1/28/24053882/arc-search-browser-web-app-ios")
- [ComfyDeploy](https://twitter.com/bennykokmusic/status/1745481428094398940?s=12&t=90xQ8sGy63D2OtiaoGJuww) - ComfyUI to API. One-click setup serverless container for each workflow, instantly scalable with @modal_labs
- Some robotics launches - [Figure-01](https://twitter.com/adcock_brett/status/1743987597301399852?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [1x studio]([https://www.youtube.com/watch?v=iHXuU3nTXfQ](https://www.youtube.com/watch?v=iHXuU3nTXfQ "https://www.youtube.com/watch?v=iHXuU3nTXfQ"))
- https://www.fixkey.ai/ - Native Apple Silicon app. Runs in the background. 0% CPU usage, Improves writing with a text replacement shortcut. openai based or [ollama](https://twitter.com/Karmedge/status/1745011856089960941)
- turbopuffer effectively soft launhced with [sam whitmore](https://twitter.com/sjwhitmore/status/1744579362782134661?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [aman sanger/@sualehasif996](https://twitter.com/amanrsanger/status/1730763587944398874)
- [Ayumi LLM RolePlay leaderboard](https://ayumi.m8geil.de/ayumi_bench_v3_results.html) - [explanation](https://rentry.co/ayumi_erp_rating)
	- [another from rentry.co with commentary](https://rentry.co/ALLMRR)
- [meals.chat](https://twitter.com/jamespotterdev/status/1752213280192643512?s=12&t=90xQ8sGy63D2OtiaoGJuww) - telegram bot for meal tracking


## misc reads 

- Model Merging
	- [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models) - a good overview from Maxime Labonne on using mergekit with 4 algorithms - SLERP, TIES, DARE, Passthrough. a simple 7B SLERP merge did well on the Open LLM Leaderboard and his own [LLM AutoEval](https://github.com/mlabonne/llm-autoeval) library
	- [HF collection of Merge paper reads from Osanseverio](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)
	- [Deepmind WARM - Weight Averaged Reward Models](https://x.com/ramealexandre/status/1749719476310843794?s=20)
	- [LLM Augmented LLMs: Expanding Capabilities through Composition](https://arxiv.org/pdf/2401.02412.pdf)- Google Research/Deepmind paper
	- https://docs.google.com/document/d/1wlG6McZzwCEFMcJsEV-hIKve2lrNFdrXTabgcvGY6z4/edit
- DPO
	- [DPO vs IPO vs KTO alternatives](https://huggingface.co/blog/pref-tuning)
		- [Main conclusion:](https://twitter.com/Teknium1/status/1748001899566235835) IPO aint it, DPO wins sometimes and KTO wins sometimes, but KTO has the advantage of not needing equivalent comparison pairs, so it's much more scalable.
- Finetuning
	- [SymNoise](https://arxiv.org/abs/2312.01523): Advancing Language Model Fine-tuning with Symmetric Noise
		- This method aims to enhance the model's function by more stringently regulating its local curvature, demonstrating superior performance over the current method, NEFTune. When fine-tuning the LLaMA-2-7B model using Alpaca, standard techniques yield a 29.79% score on AlpacaEval. However, our approach, SymNoise, increases this score significantly to 69.04%, using symmetric noisy embeddings. This is a 6.7% improvement over the state-of-the-art method, NEFTune~(64.69%). 
	- list of resources for functioncalling and tool use
		- https://x.com/BlancheMinerva/status/1748406358766968872?s=20
		- gorilla
		- toolLLM/ToolBench
- Multimodality
	- [new TTS model tracker from huggingface](https://github.com/Vaibhavs10/open-tts-tracker)
- Synthetic data
	- [Self-Rewarding LLMs](https://x.com/jaseweston/status/1748158323369611577?s=46&t=90xQ8sGy63D2OtiaoGJuww) ([HF](https://huggingface.co/papers/2401.10020), [twitter thread](https://twitter.com/jaseweston/status/1748158323369611577))
		- In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613.
		- LM itself provides its own rewards on own generations via LLM-as-a-Judge during Iterative DPO
		-   Reward modeling ability improves during training rather than staying fixed
		- ...opens the door to superhuman feedback?
		- related paper: [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)
	- [WRAP: Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/pdf/2401.16380.pdf)
		- From Apple - we propose Web Rephrase Augmented Pre-training (WRAP) that uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles such as “like Wikipedia” or in “question-answer format” to jointly pre-train LLMs on real and synthetic rephrases. First, we show that using WRAP on the C4 dataset, which is naturally noisy, speeds up pre-training by ∼ 3×. At the same pre-training compute budget, it improves perplexity by more than 10% on average across different subsets of the Pile, and improves zero-shot question answer accuracy across 13 tasks by more than 2%.
	- [Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368.pdf) Microsoft trained a text embedding model from Mistral-7B that topped the MTEB leaderboard! The secret sauce? Synthetic data. Step 1 was to generate a diverse list of retrieval tasks. [see example](https://x.com/andersonbcdefg/status/1742613575217156547?s=20)
		- We leverage proprietary LLMs to generate diverse synthetic data for hundreds of thousands of text embedding tasks across nearly 100 languages. We then fine-tune open-source decoder-only LLMs on the synthetic data using standard contrastive loss. Experiments demonstrate that our method achieves strong performance on highly competitive text embedding benchmarks without using any labeled data. F
- interesting
	- [Listening with LLM](https://paul.mou.dev/posts/2023-12-31-listening-with-llm): "the steps I took to learn how to finetune a LLM model (Mistral OpenOrca + Whisper) to describe a given audio file on Google’s MusicCaps dataset"
	- [Self-driving as a case study for AGI](https://web.archive.org/web/20240122062223/https://karpathy.github.io/2024/01/21/selfdriving-agi/)([karpathy.github.io](https://news.ycombinator.com/from?site=karpathy.github.io))
		- taken down but mirrored https://huggingface.co/posts/clem/970025506569107
	- [Building a fully local LLM voice assistant to control my smart home](https://johnthenerd.com/blog/local-llm-assistant/?utm_source=ainews&utm_medium=email)
		- I’ve had my days with Siri and Google Assistant. While they have the ability to control your devices, they cannot be customized and inherently rely on cloud services. In hopes of learning something new _and_ having something cool I could use in my life, I decided I want better.
		- The premises are simple:
			- I want my new assistant to be sassy and sarcastic.
			- I want everything running local. No exceptions. There is no reason for my coffee machine downstairs to talk to a server on the other side of the country.
			- I want more than the basic “turn on the lights” functionality. Ideally, I would like to add new capabilities in the future.
	- [Chess-GPT's Internal World Model](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html)
		- A 50 million parameter GPT trained on 5 million games of chess learns to play at ~1300 ELO in one day on 4 RTX 3090 GPUs. This model is only trained to predict the next character in PGN strings (1.e4 e5 2.Nf3 …) and is never explicitly given the state of the board or the rules of chess. Despite this, in order to better predict the next character, it learns to compute the state of the board at any point of the game, and learns a diverse set of rules, including check, checkmate, castling, en passant, promotion, pinned pieces, etc. In addition, to better predict the next character it also learns to estimate latent variables such as the ELO rating of the players in the game.
		- [Tweet thread](https://twitter.com/a_karvonen/status/1743666230127411389): "I trained Chess-GPT, a 50M parameter LLM, to play at 1500 ELO. We can visualize its internal state of the board. In addition, to better predict the next character it estimates the ELO of the players involved"
	- [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489)
		- learnings of hardware based nonlinearities in model hyperparams that cause huge efficiency jumps when optimized
		- Getting the most out of your hardware when training transformers requires thinking about your model as a sequence of GPU kernel calls. This mindset, common in HPC, is rare in ML and leads to inefficiencies in LLM training. This is especially important because LLMs tend to copy architectures from previous models. GPT-3 2.7B's architecture was used by GPT-Neo, OPT, Pythia, RedPajama-INCITE, and Cerebras-GPT, but a small tweak to its shape provides a 20% throughput improvement!
		- from eleuther team behind pythia
	- "[3 points in the AI Accelerator design space: GPU, TPU-like and spatial dataflow](https://twitter.com/nadeesha99/status/1744445585749659877?s=12&t=90xQ8sGy63D2OtiaoGJuww)" on GPU vs TPU vs Cerebras
- NIST paper: [Adversarial Machine Learning - A Taxonomy and Terminology of Attacks and Mitigations](https://twitter.com/rez0__/status/1743266573668757568)
	- link to [paper](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf)  
- Discussions:
	- Simonw vs Karpathy - [AI vs IA](https://x.com/karpathy/status/1744062845426532473?s=20)
		- followup https://twitter.com/karpathy/status/1744179910347039080
	- DPO paper got [Standing ovation from Andrew Ng](https://twitter.com/andrewyng/status/1745516258697863259?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- https://www.deeplearning.ai/the-batch/issue-231/
		- Via clever mathematical insight, the authors show that given an LLM, there is a specific reward function for which that LLM is optimal. DPO then trains the LLM directly to make the reward function (that’s now implicitly defined by the LLM) consistent with the human rankings. So you no longer need to deal with a separately represented reward function, and you can train the LLM directly to optimize the same objective as RLHF.
	- [story of the acquisition of Gradio by Huggingface](https://twitter.com/abidlabs/status/1745533306492588303?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [Facebook is aggressively going after LLaMA repos with DMCA's.](https://twitter.com/theshawwn/status/1638925249709240322)
	- [Will scaling work? - Dwarkesh](https://www.dwarkeshpatel.com/p/will-scaling-work)
		- skeptic vs believer back and forth debates
	- [Mixtral "paper" was released but contained no training details](https://twitter.com/dchaplot/status/1744547220983005478?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [Why Linear Transformers will work vs quadratic](https://x.com/jacobmbuckman/status/1744433676912345474?s=20) - basically [drop the exponential from the softmax layer](https://manifestai.com/blogposts/faster-after-all/#linear-transformers) 
	- [Nice thinking from a16z Yoko Li on LLM vs Diffusion models in creative AI](https://x.com/stuffyokodraws/status/1749834494948122977?s=20)
	- On the NYT lawsuit - [**Output similarity is a distraction. Training is the real problem.**](https://www.aisnakeoil.com/p/generative-ais-end-run-around-copyright?selection=a8f5ff68-e456-4cd0-b591-f58c03c14a7d#)
	- [Are we at peak vector database?](https://news.ycombinator.com/item?id=39119198) see comment "This is a ridiculous rant."
	- [Apple tests calling GPT 3.5 directly from device](https://x.com/ananayarora/status/1751130254046007433?s=20) and suggesting running own LLM
	- [How to do RAG with Citations](https://x.com/simonw/status/1751841193506550112?s=20)
- Learning
	- [Transformers form Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html) - Karpathy's nanoGPT worked thru a notebook
	- [Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=nOxKexn3iBo&embeds_referring_euri=https%3A%2F%2Ftwitter.com%2F&source_ve_path=MjM4NTE&feature=emb_title)
	- [Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU) - [hn recommended](https://news.ycombinator.com/item?id=38935601) 
		- [Sasha Rush's lecture as well](https://www.youtube.com/watch?v=dKJEpOtVgXc)
	- [ChatGPT at home series](https://twitter.com/NielsRogge/status/1747631048941252878): fine-tuning Mistral-7B on a GPU rented on Runpod: Involves chat templates, QLoRa, packing, Flash Attention 2, bfloat16
	- [LoRA from scratch: implementation for LLM finetuning](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?view=public&section=all)
	- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/)
		- shows how to train a control vector to add quirky traits during inference. very cool intro to representation eng
	- [Vicki Boykis on building a semantic search engine with BERT](https://vickiboykis.com/2024/01/05/retro-on-viberary/)
		- see also [Using Vectorize to build an unreasonably good search engine in 160 lines of code](https://blog.partykit.io/posts/using-vectorize-to-build-search)
	- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) using the latest research techniques, including Flash Attention, Q-LoRA, OpenAI dataset formats (messages), ChatML, Packing, all built with Hugging Face TRL
		-  for consumer-size GPUs (24GB) covering the full end-to-end lifecycle with: 
			- 💡Define and understand use cases for fine-tuning  
			- 🧑🏻‍💻 Setup of the development environment  
			- 🧮 Create and prepare dataset (OpenAI format)  
			- 🏋️‍♀️ Fine-tune LLM using TRL and the SFTTrainer  
			- 🥇 Test and evaluate the LLM  
			- 🚀 Deploy for production with TGI
		- see also [MistralTrix + DPO tutorial](https://news.ycombinator.com/item?id=38882726)
	- [Nathan Lambert with Tri Dao and Michael Poli on future of LLM architectures](https://www.youtube.com/watch?v=OFFHiJzPpCQ)
	- [The VAE used for Stable Diffusion 1.x/2.x and other models (KL-F8) has a critical flaw, probably due to bad training, that is holding back all models that use it (almost certainly including DALL-E 3)](https://old.reddit.com/r/StableDiffusion/comments/1ag5h5s/the_vae_used_for_stable_diffusion_1x2x_and_other/)
	- [Self-Driving vs AGI (deleted post)](https://huggingface.co/posts/clem/970025506569107)


## memes

- schmidhuber is right https://twitter.com/SchmidhuberAI/status/1745475698737938543
- palworld b2b ai saas https://x.com/nearcyan/status/1750056156721242212?s=20
- "it's all recsys" https://discord.com/channels/822583790773862470/839660725252784149/1202272588944904193
- ML is "hacky" [https://x.com/jim_rutt/status/1748366435913548184?s=46&t=90xQ8sGy63D2OtiaoGJuww](https://x.com/jim_rutt/status/1748366435913548184?s=46&t=90xQ8sGy63D2OtiaoGJuww "https://x.com/jim_rutt/status/1748366435913548184?s=46&t=90xQ8sGy63D2OtiaoGJuww")
- [the new captcha]([https://fxtwitter.com/rpoo/status/1740460800995963366?s=20](https://fxtwitter.com/rpoo/status/1740460800995963366?s=20 "https://fxtwitter.com/rpoo/status/1740460800995963366?s=20"))