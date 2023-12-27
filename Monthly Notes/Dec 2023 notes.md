## themes

- ML inference wars
	- mixtral price war
	- anyscale drama https://twitter.com/soumithchintala/status/1738241213327692174
		- https://buttondown.email/ainews/archive/ainews-12222023-anyscales-benchmark-criticisms/
		- https://www.anyscale.com/blog/comparing-llm-performance-introducing-the-open-source-leaderboard-for-llm
	- https://www.semianalysis.com/p/inference-race-to-the-bottom-make
	- https://vgel.me/posts/faster-inference/
	- https://pythonspeed.com/articles/cpu-thread-pool-size/
- synthetic data
	- https://arxiv.org/pdf/2312.06585.pdf karpathy pick from deepmind



## openai

- consensus on [lobotomized chatgpt](https://discord.com/channels/1168579740391710851/1168582188950896641/1182718496707203072) acknowledged by the official [twitter account](https://x.com/ChatGPTapp/status/1732979491071549792?s=20). lazier at coding - can [fix with a GPT](https://x.com/NickADobos/status/1732982713010073720?s=20)
- axelspringer partnership - [tweet](https://x.com/OpenAI/status/1734940445824937993?s=20)
- superalignment 1e7 fund
- openai [suspends bytedance for breaking TOS](https://twitter.com/alexeheath/status/1735758297893085621)
- chatgpt plus signups re enabled

## frontier models

- Google
		- Gemini
			- [32-shot chain of thought...](https://twitter.com/brickroad7/status/1732462906187325644). on 5 shot, like for like, [it is slightly worse than GPT4](https://twitter.com/_philschmid/status/1732435791358410863)
			- [BigTech LLM evals are just marketing](https://www.interconnects.ai/p/evals-are-marketing)
			- MMMU [is nice though](https://twitter.com/JeffDean/status/1732418506241790197)
			-   2 big Chinchilla / 1 small Llama (over-token) ~ 1.8B 
			- 32k Context, MQA
			- Flamingo interleaved input tokenization 
			- DALL-E 1 image output tokenization 
			- Speech (USM) and video input, no output 
			- Text benchmarks roughly eq GPT-4 
			- RLHF + Constitutional AI
			- the blogpost contains [concerning discrepancies to the video](https://twitter.com/ajones55555/status/1732609418527682709), which was [heavily edited](https://x.com/tszzl/status/1732615332471415178?s=20) - [no realtime, no voice](https://news.ycombinator.com/item?id=38559582)
				- [doesnt actually do TTS?](https://x.com/romechenko/status/1732445015123837234?s=20) 
			- [Direct comparisons with GPT4. 12/14 right](https://x.com/DimitrisPapail/status/1732529288493080600?s=20)
			- [Sergey Brin heavily contributed](https://x.com/olcan/status/1732798458615210187?s=20)
			- more videos
			- Gemini extracting relevant information from tens of thousands of scientific papers: https://youtu.be/sPiOP_CB54A
				- Highlights of the native multimodality of Gemini with audio and images: https://youtu.be/D64QD7Swr3s
				- A version of AlphaCode built on top of Gemini that performs in the top 15% of competitors in competitive programming: https://youtu.be/D64QD7Swr3s
				- Gemini helping a parent and student with their physics homework: https://youtu.be/K4pX1VAxaAI
				- Gemini creating bespoke UIs that are contextual and relevant to an ongoing conversation: https://youtu.be/v5tRc_5-8G4
				- Gemini’s approach to Responsible AI:  https://youtube.com/watch?v=gi6J_WjjNhE
				- A full set of demos is at: https://deepmind.google/gemini
			- "Gemini is a large-scale science and engineering effort, requiring all kinds of different expertise in ML, distributed systems, data, evaluation, RL, fine-tuning, and more (800+ authors on the report).  The largest Gemini model was trained on a significant number of TPUv4 pods.   It is built on top of JAX and the Pathways system (https://arxiv.org/abs/2203.12533), which enables us to orchestrate the large-scale training computation across a large number of TPUv4 pods across multiple data centers from a single Python process."
			- We have prepared a technical report about Gemini covering the model, training infrastructure, evaluations, safety analysis and responsible deployment.  I’ll walk you through some of the tables and figures in the report. https://deepmind.google/gemini/gemini_1_report.pdf
			- Gemini Pro api https://x.com/sundarpichai/status/1734952757722001626?s=20
	- [Announcing TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)
	- Meta [Emu image synthesis](https://arstechnica.com/information-technology/2023/12/metas-new-ai-image-generator-was-trained-on-1-1-billion-instagram-and-facebook-photos/)

## Models


- Mistral 8x7B
	- related paper: 
		- MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS
		- https://arxiv.org/pdf/2211.15841.pdf
	- [First AI endpoints are available in early access](https://mistral.ai/news/la-plateforme/)
		- pricing - is 4x gpt3.5turbo at $8 per mil tokens
		- 2 ~ 4 $ per 1M token for a 30B model
	- [TOS issue removed by CEO](https://twitter.com/arthurmensch/status/1734470462451732839)
	- [Mistral finetune optimized from OpenPipe](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized) calls out a few other more recent Mistral variants:
		- [OpenHermes 2.5](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B), [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [Cybertron](https://huggingface.co/fblgit/una-cybertron-7b-v2-bf16), [Intel Neural Chat](https://huggingface.co/Intel/neural-chat-7b-v3-3), [Hermes Neural](https://huggingface.co/Weyaxi/OpenHermes-2.5-neural-chat-v3-3-Slerp), and [Metamath Cybertron Starling](https://huggingface.co/Q-bert/MetaMath-Cybertron-Starling).
		- great guide on finetuning https://twitter.com/HarperSCarroll/status/1737946511856832695
	- Mixtral-instruct
		- trained with DPO
		- [avaialblel on perplxity labs](https://twitter.com/AravSrinivas/status/1734603265801613670)
	- visualizing mixtral MOE
		- https://mixtral-moe-vis-d726c4a10ef5.herokuapp.com/
		- https://news.ycombinator.com/item?id=38733208
- Nouse Hermes 2
- Apple Ferret https://appleinsider.com/articles/23/12/24/apples-ferret-is-a-new-open-source-machine-learning-model?utm_source=ainews&utm_medium=email
- Mamba models
	- [tri dao](https://twitter.com/tri_dao/status/1731728602230890895) and [albert gu](https://twitter.com/_albertgu/status/1731727672286294400)
	- [state space models due to "selection" mechanism](https://x.com/IntuitMachine/status/1732055797788528978?s=20)
	- notable performance for [130m models](https://x.com/__vec__/status/1732603830817198228?s=20)
	- [outside of pytorch](https://twitter.com/srush_nlp/status/1731751599305879593)
	- [Mamba chat - finetuned for chat](https://x.com/MatternJustus/status/1732572463257539032?s=20)
- Phi-2
	- https://x.com/sytelus/status/1734881560271454525?s=20
	- ehartford version of it https://twitter.com/erhartford/status/1738677760200155464
- StripedHyena
	- descendant of [Hyena](https://arxiv.org/abs/2302.10866)
- https://ai.meta.com/research/seamless-communication/
	- SeamlessExpressive: A model that aims to preserve expression and intricacies of speech across languages.
	- SeamlessStreaming: A model that can deliver speech and text translations with around two seconds of latency.
	- SeamlessM4T v2: A foundational multilingual and multitask model that allows people to communicate effortlessly through speech and text.
	- Seamless: A model that merges capabilities from SeamlessExpressive, SeamlessStreaming and SeamlessM4T v2 into one.
- [Magicoder: Source Code Is All You Need](https://arxiv.org/abs/2312.02120)
	- We introduce Magicoder, a series of fully open-source (code, weights, and data) Large Language Models (LLMs) for code that significantly closes the gap with top code models while having no more than 7B parameters. Magicoder models are trained on 75K synthetic instruction data using OSS-Instruct, a novel approach to enlightening LLMs with open-source code snippets to generate high-quality instruction data for code. 
	- The orthogonality of OSS-Instruct and other data generation methods like Evol-Instruct further enables us to build an enhanced MagicoderS. 
	- Notably, MagicoderS-CL-7B based on CodeLlama even surpasses the prominent ChatGPT on HumanEval+ (66.5 vs. 65.9 in pass@1). Overall, OSS-Instruct opens a new direction for low-bias and high-quality instruction tuning using abundant open-source references.
- [NexusRaven-v2 13b for function calling LLM for GPT4 zero shot tool use](https://x.com/togethercompute/status/1732092331581636875?s=20)
	- NexusRaven V2 was instruction-tuned from @AIatMeta 's CodeLlama-13B, without using proprietary LLM generated data.

- google imagen 2 https://news.ycombinator.com/item?id=38628417
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](https://jingyechen.github.io/textdiffuser2/)
	- solves text-in-images, including inpainting text
	- "Firstly, we fine-tune a large language model for layout planning. The large language model is capable of automatically generating keywords for text rendering and also supports layout modification through chatting. Secondly, we utilize the language model within the diffusion model to encode the position and texts at the line level. Unlike previous methods that employed tight character-level guidance, this approach generates more diverse text images."
- [LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/abs/2312.06550)
	- We present LLM360, an initiative to fully open-source LLMs, which advocates for all training code and data, model checkpoints, and intermediate results to be made available to the community. The goal of LLM360 is to support open and collaborative AI research by making the end-to-end LLM training process transparent and reproducible by everyone. As a first step of LLM360, we release two 7B parameter LLMs pre-trained from scratch, Amber and CrystalCoder, including their training code, data, intermediate checkpoints, and analyses (at [this https URL](https://www.llm360.ai/)).


## open source tooling and projects

- https://github.com/monoidspace/monoid **Turn your APIs into AI Agents**
	- 🔌 Plug and play with different LLMs and Agent Types with the click of a button
	- 📬 Postman-like interface for turning your APIs into Actions, where you can choose which parameters the Agent controls
	- 🏖️ Action Sandbox to "talk" to your API in natural language, where you can simulate an Agent who only has one Action
	- 🤖 Agent Sandbox to simulate and test your AI Agent before you deploy it
	- 🪆 Use Agents as Actions within other Agents, so that they can collaborate and solve more complex problems
	- 🤝 Action Hub and Agent Hub to allow the community to share its creations and build off each other's work
- https://github.com/CopilotKit/CopilotKit
	- <CopilotPortal />: Build in-app AI chatbots that can "see" the current app state + take action inside your app. The AI chatbot can talk to your app frontend & backend, and to 3rd party services (Salesforce, Dropbox, etc.) via plugins.
	- <CopilotTextarea />: AI-assisted text generation. Drop-in replacement for any <textarea />. Autocompletions + AI editing + generate from scratch. Indexed on your users' content.
- https://postgresml.org/blog/introducing-the-openai-switch-kit-move-from-closed-to-open-source-ai-in-minutes
	- an open-source AI SDK (Python & JavaScript) that provides a drop-in replacement for OpenAI’s chat completion endpoint. We'd love to know what you think so we can make switching as easy as possible and get more folks on open-source.
- voice cloning with oss models https://replicate.com/blog/how-to-tune-a-realistic-voice-clone
- https://github.com/turboderp/exllamav2
	- ollama alternative
- [open source macos copilot](https://news.ycombinator.com/item?id=38611700)
- namedrop
	- https://twitter.com/charliebholtz/status/1737667912784134344
	- https://github.com/cbh123/namedrop
	- ollama-namedrop

You can swap in almost any open-source model on Huggingface. HuggingFaceH4/zephyr-7b-beta, Gryphe/MythoMax-L2-13b, teknium/OpenHermes-2.5-Mistral-7B and more.

- autogen added a new UI layer https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-assistant

## fundraising

- [Anthropic 750m @ 15b valuation](https://www.theinformation.com/articles/anthropic-to-raise-750-million-in-menlo-ventures-led-deal)
- OpenAI at 100b valuation
- [Mistral 400m @ 2b valuation](https://twitter.com/abacaj/status/1733262949475623142/photo/1)
- [AssemblyAI 50m series B?](https://techcrunch.com/2023/12/04/assemblyai-nabs-50m-to-build-and-serve-ai-speech-models/)
	- AssemblyAI claims that its paying customer base grew 200% from last year to 4,000 brands and that its AI platform is now handling around 25 million API calls per day. Moreover, over 200,000 developers are building on the platform, AssemblyAI says — using it to process more than 10 terabytes of data a day.
	- A slice of the new funding will be put toward a “universal speech model that the company’s training on over a petabyte of voice data, set to launch later this year,” Fox says. AssemblyAI is also expanding its headcount, aiming to grow its 115-person workforce by 50% to 75% next year
- [replicate 40m series B](https://twitter.com/replicate/status/1732104158877188305)
- [leonardo ai $31m Series A](https://techcrunch.com/2023/12/06/leonardo-ai/)
- [extropic ai 14m seed](https://twitter.com/Extropic_AI/status/1731675230513639757)
- martian fundraise announced

## other launhces

- [Bing Code Interpreter for free!](https://twitter.com/MParakhin/status/1732094937368494280)
- Lume, a seed-stage startup ([https://www.lume.ai/](https://www.lume.ai/)): use AI to automatically transform your source data into any desired target schema in seconds, making onboarding client data or integrating with new systems take seconds rather than days or weeks. In other words, we use AI to automatically map data between any two data schemas, and output the transformed data to you.
- [1 year anniversary of perplexity ai](https://x.com/AravSrinivas/status/1732825206023201273?s=20)
- suno ai music generation 
	- https://twitter.com/sjwhitmore/status/1737569171960209452
	- https://twitter.com/karpathy/status/1737518588159041845

## misc discussions and reads

- [Fine Tuning Mistral 7B on Magic the Gathering Drafts](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the)
- [Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://qtransformer.github.io/)
- [Jailbroken AI Chatbots Can Jailbreak Other Chatbots](https://www.scientificamerican.com/article/jailbroken-ai-chatbots-can-jailbreak-other-chatbots/)
AI chatbots can convince other chatbots to instruct users how to build bombs and cook meth
- [Distilwhisper explainer](https://twitter.com/srush_nlp/status/1737837726572150851)
- pydantic is all you need https://minimaxir.com/2023/12/chatgpt-structured-data/
	- including for chain of thought!
- [How to make LLMs go fast](https://vgel.me/posts/faster-inference/)
- [LoftQ - drop-in QLoRA replacement](https://x.com/WeizhuChen/status/1736127441238913438?s=20)
- [Benchmarknig function calling](https://twitter.com/robertnishihara/status/1734629320868687991)  https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features
	⚫️ gpt-4: 93.00 ± 0.00
	⚫️ mistral-7b: 81.50 ± 0.96
	⚫️ llama-2-70b: 81.00 ± 0.41
	⚫️ gpt-3.5-turbo: 81.00 ± 1.47
	⚫️ llama-2-13b: 79.75 ± 0.63
	⚫️ zephyr-7b-beta: 70.50 ± 0.87
	⚫️ llama-2-7b: 60.75 ± 1.31


## memes

- decent safety meme https://fxtwitter.com/bitcloud/status/1731974050681909714?s=20