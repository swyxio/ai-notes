
## openai

- OpenAI Sora - [official blog](https://openai.com/sora)
	- [lead author credits](https://x.com/sama/status/1758193609927721350?s=20)
	- [sama showing demos in response to user prompts](https://x.com/sama/status/1758193792778404192?s=20)
- ChatGPT got new memory and new controls https://news.ycombinator.com/item?id=39360724
	- [uses a new bio tool](https://x.com/simonw/status/1757629263338209584?s=20)
- gpt-3.5-turbo-0125
	- The updated GPT-3.5 Turbo model is now available. It comes with 50% reduced input pricing, 25% reduced output pricing, along with various improvements including higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Returns a maximum of 4,096 output tokens.
- [chatgpt in apple vision pro](https://x.com/ChatGPTapp/status/1753480051889508485?s=20)
- lazy openai
	- https://www.reddit.com/r/OpenAI/comments/1aj6lrz/damned_lazy_ai/
- nontechnical
	- shut down [State-affiliated Threat Actors](https://openai.com/blog/disrupting-malicious-uses-of-ai-by-state-affiliated-threat-actors)
- public appearances
	- logan on a big pod today https://www.lennyspodcast.com/inside-openai-logan-kilpatrick-head-of-developer-relations/

Frontier models
- RIP Bard [https://twitter.com/AndrewCurran_/status/1754546359460590002](https://twitter.com/AndrewCurran_/status/1754546359460590002 "https://twitter.com/AndrewCurran_/status/1754546359460590002")
	- https://blog.google/products/gemini/bard-gemini-advanced-app/
- Gemini ultra 1.0
	- [unclear advantage over gemini pro](https://www.youtube.com/watch?v=hLbIUQWxs6Y)
- Gemini 1.5
	- 1m context supported, 10m capacity in research
	- [official blogpost](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15), [technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf), [hn](https://news.ycombinator.com/item?id=39383446)
		- This includes making Gemini 1.5 more efficient to train and serve, with a new [Mixture-of-Experts](https://arxiv.org/abs/1701.06538) (MoE) architecture.
		- The first Gemini 1.5 model we’re releasing for early testing is Gemini 1.5 Pro. It’s a mid-size multimodal model, optimized for scaling across a wide-range of tasks, and [performs at a similar level to 1.0 Ultra](https://goo.gle/GeminiV1-5), our largest model to date. It also introduces a breakthrough experimental feature in long-context understanding.
		- **Gemini 1.5 Pro comes with a standard 128,000 token context window. But starting today, a limited group of developers and enterprise customers can try it with a context window of up to 1 million tokens via [AI Studio](https://aistudio.google.com/) and [Vertex AI](https://cloud.google.com/vertex-ai) in private preview.**
		- Through a series of machine learning innovations, we’ve increased 1.5 Pro’s context window capacity far beyond the original 32,000 tokens for Gemini 1.0. We can now run up to 1 million tokens in production.
		- when tested on long code prompts, [HVM author agrees gemini retrieval is better than GPT4](https://old.reddit.com/r/singularity/comments/1atjz9v/ive_put_a_complex_codebase_into_a_single/)
	-  Sparse MoE multimodal model - Can handle 3 hours of video, 22 hours of audio or 10M tokens with almost perfect recall! - Better than Gemini 1 Pro, better than Ultra for text, worse for audio and vision - Sadly not much info regarding model size, # of experts, architecture explorations, etc
- https://twitter.com/evowizz/status/1753795479543132248

## models
- Cohere AI for good - Aya 101 - very good multilingual model outperforming BLOOM despite 2x languages. Apache 2.0 license
- [Stable Cascade](https://news.ycombinator.com/item?id=39360106): a new arch vs stable diffusion
	-  Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a 1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the highly compressed latent space. 
	- Previous versions of this architecture, achieved a 16x cost reduction over Stable Diffusion 1.5.
	- Stable Cascade consists of three models: Stage A, Stage B and Stage C, representing a cascade for generating images, hence the name "Stable Cascade". Stage A & B are used to compress images, similarly to what the job of the VAE is in Stable Diffusion. However, as mentioned before, with this setup a much higher compression of images can be achieved. Furthermore, Stage C is responsible for generating the small 24 x 24 latents given a text prompt. The following picture shows this visually. Note that Stage A is a VAE and both Stage B & C are diffusion models.
	- For this release, we are providing two checkpoints for Stage C, two for Stage B and one for Stage A. Stage C comes with a 1 billion and 3.6 billion parameter version, but we highly recommend using the 3.6 billion version, as most work was put into its finetuning. The two versions for Stage B amount to 700 million and 1.5 billion parameters. Both achieve great results, however the 1.5 billion excels at reconstructing small and fine details. Therefore, you will achieve the best results if you use the larger variant of each. Lastly, Stage A contains 20 million parameters and is fixed due to its small size.
- [Nomic Embed](https://twitter.com/nomic_ai/status/1753082063048040829): Open source, open weights, open data
	- https://blog.nomic.ai/posts/nomic-embed-text-v1
	- Beats OpenAI text-embeding-3-small and Ada on short and long context benchmarks
	- [nomic 1.5](https://x.com/xenovacom/status/1757798436009599413?s=46&t=90xQ8sGy63D2OtiaoGJuww) wiht MRL embeddings and 8192 context
- AI OLMo - 100% open-everything model
	- https://blog.allenai.org/olmo-open-language-model-87ccfc95f580
	- [released on magnet link](https://twitter.com/natolambert/status/1753063313351835941)
- [Natural-SQL-7B, a strong text-to-SQL model](https://github.com/cfahlgren1/natural-sql)
- [DeepSeekMath 7B](https://arxiv.org/abs/2402.03300) which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH
- [Presenting MetaVoice-1B](https://twitter.com/metavoiceio/status/1754983953193218193), a 1.2B parameter base model for TTS (text-to-speech). Trained on 100K hours of data. * Emotional speech in English * Voice cloning with fine-tuning * Zero-shot cloning for American & British voices * Support for long-form synthesis. Best part: Apache 2.0 licensed. 🔥
	- https://ttsdemo.themetavoice.xyz/
- [Reka Flash](https://twitter.com/YiTayML/status/1757115386829619534), a new state-of-the-art 21B multimodal model that rivals Gemini Pro and GPT 3.5 on key language & vision benchmarks 
	- [live chatted with Yi Tay and Max on ThursdAI](https://twitter.com/altryne/status/1758181289218490605) (1hr in)
- [SPIRIT-LM: Interleaved Spoken and Written Language Model](https://speechbot.github.io/spiritlm/index.html) from Meta
	- compare with: [LAION BUD-E](https://laion.ai/blog/bud-e/) - ENHANCING AI VOICE ASSISTANTS’ CONVERSATIONAL QUALITY, NATURALNESS AND EMPATHY
		- Right now (January 2024) we reach latencies of 300 to 500 ms (with a Phi 2 model). We are confident that response times below 300 ms are possible even with larger models like LLama 2 30B in the near future.


## open source tooling and projects

- [Ollama openai compatibility APIs](https://news.ycombinator.com/item?id=39307330)
- [LlamaIndex v0.10](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8?source=collection_home---6------0-----------------------)
- [OpenLLMetry-JS](https://news.ycombinator.com/item?id=39371297) SDK via traceloop
	- an open protocol and SDK, based on OpenTelemetry, that provides traces and metrics for LLM JS/TS applications and can be connected to any of the 15+ tools that already support OpenTelemetry.
	- JS version of [Python one](https://news.ycombinator.com/item?id=37843907)
- [Reor – An AI note-taking app that runs models locally](https://github.com/reorproject/reor)

## product launches

- vercel ai integrations - aggregation theory on AI


## Misc reads

- discussions
	- [Umichigan selling student data](https://x.com/suchenzang/status/1758020313689006374?s=20) 
	- [Interesting discussion on Replicate's poor GPU cold start with CEO](https://news.ycombinator.com/item?id=39411748)
- learning
	- **[TPU-Alignment](https://github.com/Locutusque/TPU-Alignment)** - Fully fine-tune large models like Mistral-7B, Llama-2-13B, or Qwen-14B completely for free. on the weekly 20hrs of TPUv3-8 pod from Kaggle 
	- [undo llama2 safety tuning with $200 LoRA](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?)
	- 

## fundraising

- Langchain $25m series A with sequoia
	- and [LangSmith GA](https://x.com/hwchase17/status/1758170252272418978?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- covered in [Forbes](https://www.forbes.com/sites/alexkonrad/2024/02/15/open-source-ai-startup-langchain-launches-langsmith/?sh=42226ad64f00)
	- Since launching LangSmith, we’ve seen:
		- Over 80K signups
		- Over 5K monthly active teams
		- Over 40 million traces logged in January alone
	- features
		- prototyping: **Initial Test Set**, **Comparison View**, **Playground**
		- beta testing: **Collecting Feedback**, **Annotating Traces**, **Adding Runs to a Dataset**
		- production: **Monitoring and A/B Testing**, 
	-  future directions include:
		- Ability to run online evaluators on a sample of production data
		- Better filtering and conversation support
		- Easy deployment of applications with hosted LangServe
		- Enterprise features to support the administration and security needs for our largest customers.
- [Magic.dev $117m](https://twitter.com/magicailabs/status/1758140204446323188) series A [with Nat and Dan](https://twitter.com/natfriedman/status/1758143612561568047)
	- "Code generation is both a product and a path to AGI, requiring new algorithms, lots of CUDA, frontier-scale training, RL, and a new UI."
	- [talking about "many millions of tokens" context](https://x.com/Hersh_Desai/status/1758147122631757829?s=20) now but previously talked about [5m tokens for "LTM-1"](https://magic.dev/blog/ltm-1)
	- nat made their own evals
- Lambda Labs $320m series C ([twitter](https://x.com/stephenbalaban/status/1758154395412214248?s=46&t=90xQ8sGy63D2OtiaoGJuww)) with USIT
	- This new financing will be used to expand the number of NVIDIA GPUs available in Lambda Cloud and build features that will absolutely delight you.


## memes

- https://twitter.com/JackPosobiec/status/1753416551066181672 dignifAI
- https://www.goody2.ai/chat
- image of no elephant https://www.reddit.com/r/OpenAI/comments/1anm3p3/damn_sneaky/
	- source of meme: https://twitter.com/GaryMarcus/status/1755476468157833593
- apple vision pro https://discord.com/channels/822583790773862470/839660725252784149/1204047864096096328
- chatgpt memory x tipping meme https://x.com/_mira___mira_/status/1757695161671565315?s=46&t=90xQ8sGy63D2OtiaoGJuww
- karpathy leaving meme https://x.com/yacinemtb/status/1757775133761057113?s=46&t=90xQ8sGy63D2OtiaoGJuww