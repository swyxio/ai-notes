
## openai

- consensus on [lobotomized chatgpt](https://discord.com/channels/1168579740391710851/1168582188950896641/1182718496707203072) acknowledged by the official [twitter account](https://x.com/ChatGPTapp/status/1732979491071549792?s=20). lazier at coding - can [fix with a GPT](https://x.com/NickADobos/status/1732982713010073720?s=20)

## frontier models

- Google
		- Gemini
			- [32-shot chain of thought...](https://twitter.com/brickroad7/status/1732462906187325644). on 5 shot, like for like, [it is slightly worse than GPT4](https://twitter.com/_philschmid/status/1732435791358410863)
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
	- [Announcing TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)
	- Meta [Emu image synthesis](https://arstechnica.com/information-technology/2023/12/metas-new-ai-image-generator-was-trained-on-1-1-billion-instagram-and-facebook-photos/)

## Models


- Mistral 8x7B
	- MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS
	- https://arxiv.org/pdf/2211.15841.pdf
- Mamba models
	- [tri dao](https://twitter.com/tri_dao/status/1731728602230890895) and [albert gu](https://twitter.com/_albertgu/status/1731727672286294400)
	- [state space models due to "selection" mechanism](https://x.com/IntuitMachine/status/1732055797788528978?s=20)
	- notable performance for [130m models](https://x.com/__vec__/status/1732603830817198228?s=20)
	- [outside of pytorch](https://twitter.com/srush_nlp/status/1731751599305879593)
	- [Mamba chat - finetuned for chat](https://x.com/MatternJustus/status/1732572463257539032?s=20)
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

You can swap in almost any open-source model on Huggingface. HuggingFaceH4/zephyr-7b-beta, Gryphe/MythoMax-L2-13b, teknium/OpenHermes-2.5-Mistral-7B and more.


## fundraising

- [Mistral 400m @ 2b valuation](https://twitter.com/abacaj/status/1733262949475623142/photo/1)
- [replicate 40m series B](https://twitter.com/replicate/status/1732104158877188305)
- [extropic ai 14m seed](https://twitter.com/Extropic_AI/status/1731675230513639757)

## other launhces

- [Bing Code Interpreter for free!](https://twitter.com/MParakhin/status/1732094937368494280)
- Lume, a seed-stage startup ([https://www.lume.ai/](https://www.lume.ai/)): use AI to automatically transform your source data into any desired target schema in seconds, making onboarding client data or integrating with new systems take seconds rather than days or weeks. In other words, we use AI to automatically map data between any two data schemas, and output the transformed data to you.
- [1 year anniversary of perplexity ai](https://x.com/AravSrinivas/status/1732825206023201273?s=20)


## misc discussions and reads

- [Fine Tuning Mistral 7B on Magic the Gathering Drafts](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the)
- [Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://qtransformer.github.io/)
- [Jailbroken AI Chatbots Can Jailbreak Other Chatbots](https://www.scientificamerican.com/article/jailbroken-ai-chatbots-can-jailbreak-other-chatbots/)
AI chatbots can convince other chatbots to instruct users how to build bombs and cook meth


## memes

- decent safety meme https://fxtwitter.com/bitcloud/status/1731974050681909714?s=20