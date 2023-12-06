
## frontier models

- Google
		- Gemini
			- [32-shot chain of thought...](https://twitter.com/brickroad7/status/1732462906187325644). on 5 shot, like for like, [it is slightly worse than GPT4](https://twitter.com/_philschmid/status/1732435791358410863)
			-   2 big Chinchilla / 1 small Llama (over-token) ~ 1.8B 
			- 32k Context, MQA
			- Flamingo interleaved input tokenization 
			- DALL-E 1 image output tokenization 
			- Speech (USM) and video input, no output 
			- Text benchmarks roughly eq GPT-4 
			- RLHF + Constitutional AI
	- [Announcing TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)

## Models


- https://ai.meta.com/research/seamless-communication/
	- SeamlessExpressive: A model that aims to preserve expression and intricacies of speech across languages.
	- SeamlessStreaming: A model that can deliver speech and text translations with around two seconds of latency.
	- SeamlessM4T v2: A foundational multilingual and multitask model that allows people to communicate effortlessly through speech and text.
	- Seamless: A model that merges capabilities from SeamlessExpressive, SeamlessStreaming and SeamlessM4T v2 into one.
- [Magicoder: Source Code Is All You Need](https://arxiv.org/abs/2312.02120)
	- We introduce Magicoder, a series of fully open-source (code, weights, and data) Large Language Models (LLMs) for code that significantly closes the gap with top code models while having no more than 7B parameters. Magicoder models are trained on 75K synthetic instruction data using OSS-Instruct, a novel approach to enlightening LLMs with open-source code snippets to generate high-quality instruction data for code. 
	- The orthogonality of OSS-Instruct and other data generation methods like Evol-Instruct further enables us to build an enhanced MagicoderS. 
	- Notably, MagicoderS-CL-7B based on CodeLlama even surpasses the prominent ChatGPT on HumanEval+ (66.5 vs. 65.9 in pass@1). Overall, OSS-Instruct opens a new direction for low-bias and high-quality instruction tuning using abundant open-source references.

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

You can swap in almost any open-source model on Huggingface. HuggingFaceH4/zephyr-7b-beta, Gryphe/MythoMax-L2-13b, teknium/OpenHermes-2.5-Mistral-7B and more.


## fundraising

- [replicate 40m series B](https://twitter.com/replicate/status/1732104158877188305)
- [extropic ai 14m seed](https://twitter.com/Extropic_AI/status/1731675230513639757)

## other launhces

- [Bing Code Interpreter for free!](https://twitter.com/MParakhin/status/1732094937368494280)
- 


## misc discussions and reads

- [Fine Tuning Mistral 7B on Magic the Gathering Drafts](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the)
- [Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://qtransformer.github.io/)
- [Jailbroken AI Chatbots Can Jailbreak Other Chatbots](https://www.scientificamerican.com/article/jailbroken-ai-chatbots-can-jailbreak-other-chatbots/)
AI chatbots can convince other chatbots to instruct users how to build bombs and cook meth


## memes

- decent safety meme https://fxtwitter.com/bitcloud/status/1731974050681909714?s=20