## Synthetic data

- Synthetic data ['is the future'](https://huggingface.co/datasets/andersonbcdefg/synthetic_retrieval_tasks) from Nous Research founder
- Microsoft trained a text embedding model from Mistral-7B that topped the MTEB  leaderboard [using synthetic data](https://twitter.com/andersonbcdefg/status/1742613575217156547)


## openai

- gpt store launch
	- [pending](https://news.ycombinator.com/item?id=38870249) , [launched on Jan 10](https://twitter.com/sama/status/1745135061731803571)
	- new docs for [Actions for GPTS](https://platform.openai.com/docs/actions/introduction)
	- [featured GPTS](https://x.com/chatgptapp/status/1750316948444086697)
- launched and rolled back [GPT personalization and Temporary Chat](https://x.com/AndrewCurran_/status/1744923452572852608?s=20)
- [GPT builder is itself a GPT](https://help.openai.com/en/articles/8770868-gpt-builder)
- [API key based usage tracking](https://twitter.com/OfficialLoganK/status/1743401083920097432) - just tracking, no limits yet
- [removed usage policy against "military and warfare" use](https://theintercept.com/2024/01/12/open-ai-military-ban-chatgpt/)
- announced new [preparedness/safety framework](https://openai.com/safety/preparedness)
- sama tsmc https://x.com/basedbeffjezos/status/1748903066735354030?s=46&t=90xQ8sGy63D2OtiaoGJuww
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


## anthropic

- [sleeper agents](https://fxtwitter.com/AnthropicAI/status/1745854907968880970)

## fundraising

- [ElevenLabs series B](https://twitter.com/elevenlabsio/status/1749435751656231065) - 80m at 1b valuation with a16z and nat/dan
	- [seed pitch deck](https://x.com/chiefaioffice/status/1749385259517440495?s=20)
- [perplexity series b](https://twitter.com/perplexity_ai/status/1742915781690798290) - 74m at 520m valuation
	- 10m MAU, 500m queries in 2023
	- https://blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round
	- https://www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger-using-ai-to-try-to-upend-internet-search-0859bda6?mod=hp_lead_pos4
- [Quora raised $75m from a16z](https://x.com/adamdangelo/status/1744805602436825334?s=20): "his funding will be used to accelerate the growth of Poe, and we expect the majority of it to be used to pay bot creators through our recently-launched creator monetization program."
- 


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
	- JSONalyze https://docs.llamaindex.ai/en/latest/examples/query_engine/JSONalyze_query_engine.html#
- open interpreter 0.2 - vision models, and an api
	- https://twitter.com/hellokillian/status/1743418943120040109
	- https://api.openinterpreter.com/ The Open Interpreter Project has developed (and here freely hosts) an API which is capable of locating visual controls with single-pixel precision.
- Codium - [AlphaCodium](https://www.codium.ai/blog/alphacodium-state-of-the-art-code-generation-for-code-contests/) - beating deepmind alphacoder/deepseek/gpt4 with prompt flow  
	- [itamar's intro video](https://twitter.com/itamar_mar/status/1747957348293824676)
	- https://x.com/karpathy/status/1748043513156272416?s=20
		- Prompt engineering (or rather "Flow engineering") intensifies for code generation. Great reading and a reminder of how much alpha there is (pass@5 19% to 44%) in moving from a naive prompt:answer paradigm to a "flow" paradigm, where the answer is constructed iteratively.
	- https://twitter.com/svpino/status/1747971746047627682
	- https://twitter.com/swyx/status/1748084170537291923


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
- [Lumiere](https://lumiere-video.github.io/): A Space-Time Diffusion Model for Realistic Video Generation

## other launches

- [Rabbit R-1 $200 LLM smartphone launched at CES](https://news.ycombinator.com/item?id=38930126)
	- highly produced 30min video, product seems polished, but people mostly unconvinced this needs to be a separate device from phone.
	- [open source clone effort with openinterpreter](https://twitter.com/hellokillian/status/1745875973583896950)
- [Together Embeddings](https://x.com/togethercompute/status/1745500191103553794?s=20)
	- just serving leading oss models incl M2-BERT 32k, UAE-large-v1 and BGE-Base-En-v1.5
	- 3 new embeddings models using Monarch Mixer architecture, enabling long context embeddings up to 32k!
	- <4x lower pricing than OpenAI/Cohere
	- integrations with MongoDB Atlas, Langchain, Llamaindex
-  [artificialanalysis.ai](https://news.ycombinator.com/item?id=39014985) - Benchmarks and comparison of LLM AI models and API hosting providers - [swyx tweet](https://twitter.com/swyx/status/1747741795281412133)
	- Martian also launched [leaderboard](https://leaderboard.withmartian.com) but not as good 
- Nightshade/Glaze 1.0 ([tweet](https://twitter.com/alexjc/status/1748754290435395739), [podcast](https://twimlai.com/podcast/twimlai/nightshade-data-poisoning-to-fight-generative-ai/))
	- a tool that turns any image into a data sample that is unsuitable for model training. More precisely, Nightshade transforms images into "poison" samples, so that models training on them without consent will see their models learn unpredictable behaviors that deviate from expected norms, e.g. a prompt that asks for an image of a cow flying in space might instead get an image of a handbag floating in space.
	- [circumventing Nightshade is violation of DMCA](https://twitter.com/alexjc/status/1748754290435395739)
- Vx.dev
	- github actions-driven development
	- vx.dev was initially designed as an open-source alternative to Vercel's v0.dev, so we also have a [blog](https://step-saga-examples.pages.dev/v0-dev-reverse-engineer/) about how we reverse-engineered v0.dev.
	- also has an interesting blogpost on [How I Reverse Engineered Vercel's v0.dev Prompt and Code Optimization Logic](https://step-saga-examples.pages.dev/v0-dev-reverse-engineer/)
- Krea.ai launching Portrait, Concept CGI and Cartoon modes
	- [Cerebral Valley interview](https://cerebralvalley.beehiiv.com/p/krea-building-next-frontier-human-creativity)

## misc reads 

- Model Merging
	- https://huggingface.co/blog/mlabonne/merge-models
	- https://docs.google.com/document/d/1wlG6McZzwCEFMcJsEV-hIKve2lrNFdrXTabgcvGY6z4/edit
- DPO
	- [DPO vs IPO vs KTO alternatives](https://huggingface.co/blog/pref-tuning)
		- [Main conclusion:](https://twitter.com/Teknium1/status/1748001899566235835) IPO aint it, DPO wins sometimes and KTO wins sometimes, but KTO has the advantage of not needing equivalent comparison pairs, so it's much more scalable.
- Multimodality
	- [new TTS model tracker from huggingface](https://github.com/Vaibhavs10/open-tts-tracker)
- [Self-Rewarding LLMs](https://x.com/jaseweston/status/1748158323369611577?s=46&t=90xQ8sGy63D2OtiaoGJuww) ([HF](https://huggingface.co/papers/2401.10020))
	- In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613.
	- LM itself provides its own rewards on own generations via LLM-as-a-Judge during Iterative DPO
	-   Reward modeling ability improves during training rather than staying fixed
	- ...opens the door to superhuman feedback?
- interesting/interpretability
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
- Learning
	- [ChatGPT at home series](https://twitter.com/NielsRogge/status/1747631048941252878): fine-tuning Mistral-7B on a GPU rented on Runpod: Involves chat templates, QLoRa, packing, Flash Attention 2, bfloat16
	- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) using the latest research techniques, including Flash Attention, Q-LoRA, OpenAI dataset formats (messages), ChatML, Packing, all built with Hugging Face TRL
		-  for consumer-size GPUs (24GB) covering the full end-to-end lifecycle with: 
			- 💡Define and understand use cases for fine-tuning  
			- 🧑🏻‍💻 Setup of the development environment  
			- 🧮 Create and prepare dataset (OpenAI format)  
			- 🏋️‍♀️ Fine-tune LLM using TRL and the SFTTrainer  
			- 🥇 Test and evaluate the LLM  
			- 🚀 Deploy for production with TGI


## memes

- schmidhuber is right https://twitter.com/SchmidhuberAI/status/1745475698737938543
- palworld b2b ai saas https://x.com/nearcyan/status/1750056156721242212?s=20