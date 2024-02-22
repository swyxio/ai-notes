
## potential themes of the month

- video
	- sora
	- [gemini 1.5](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)
- long context vs rag

## openai

- OpenAI Sora - [official blog](https://openai.com/sora)
	- [lead author credits](https://x.com/sama/status/1758193609927721350?s=20)
	- [sama showing demos in response to user prompts](https://x.com/sama/status/1758193792778404192?s=20)
		- and on openai tiktok ([dog](https://x.com/venturetwins/status/1759984752206196961?s=20), [pizza](https://www.tiktok.com/@openai/video/7337782565870357803?_r=1&_t=8k3QSuBFQhW))
	- ylecun hating on world model analogies as usual [https://twitter.com/ylecun/status/1759486703696318935](https://twitter.com/ylecun/status/1759486703696318935 "https://twitter.com/ylecun/status/1759486703696318935")
		- Modeling the world for action by generating pixel is as wasteful and doomed to failure as the largely-abandoned idea of "analysis by synthesis".
		- If your goal is to train a world model for recognition or planning, using pixel-level prediction is a terrible idea. Generation happens to work for text because text is discrete with a finite number of symbols. Dealing with uncertainty in the prediction is easy in such settings. Dealing with prediction uncertainty in high-dimension continuous sensory inputs is simply intractable. That's why generative models for sensory inputs are doomed to failure.
- ChatGPT got new memory and new controls https://news.ycombinator.com/item?id=39360724
	- [uses a new bio tool](https://x.com/simonw/status/1757629263338209584?s=20)
- gpt-3.5-turbo-0125
	- The updated GPT-3.5 Turbo model is now available. It comes with 50% reduced input pricing, 25% reduced output pricing, along with various improvements including higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Returns a maximum of 4,096 output tokens.
- gpt4turbo 0125 has training data updated to dec 2023
- [chatgpt in apple vision pro](https://x.com/ChatGPTapp/status/1753480051889508485?s=20)
- chatgpt bugs and flaws
	- https://www.reddit.com/r/OpenAI/comments/1aj6lrz/damned_lazy_ai/
	- ChatGPT went "[beserk](https://garymarcus.substack.com/p/chatgpt-has-gone-berserk)" on Feb 20. 
		- [going off the rails](https://twitter.com/seanw_m/status/1760115118690509168), [speaking Spanglish](https://twitter.com/seanw_m/status/1760115732333941148)
		- Acknowledged [by OpenAI status page](https://twitter.com/seanw_m/status/1760116061116969294)
		- [fixed with reason](https://twitter.com/E0M/status/1760476148763644166)
	- [due to lazy/extensive prompt?](https://twitter.com/dylan522p/status/1755086111397863777)
- nontechnical
	- shut down [State-affiliated Threat Actors](https://openai.com/blog/disrupting-malicious-uses-of-ai-by-state-affiliated-threat-actors)
- public appearances
	- logan on a big pod today https://www.lennyspodcast.com/inside-openai-logan-kilpatrick-head-of-developer-relations/

## Frontier models

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
	- useful for [scanning codebase and implementing fixes for](https://x.com/sullyomarr/status/1760066335898513655?s=46&t=90xQ8sGy63D2OtiaoGJuww) and 
- [Gemini rollout in Google Workspace (Gmail, docs, sheets)](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/)
	- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)
- https://twitter.com/evowizz/status/1753795479543132248

## models
- [Meta V-JEPA model](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/?utm_source=ainews&utm_medium=email) - video version of I-JEPA model
	- V-JEPA is a non-generative model that learns by predicting missing or masked parts of a video in an abstract representation space. This is similar to how our [Image Joint Embedding Predictive Architecture (I-JEPA)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) compares abstract representations of images (rather than comparing the pixels themselves). Unlike generative approaches that try to fill in every missing pixel, V-JEPA has the flexibility to discard unpredictable information, which leads to improved training and sample efficiency by a factor between 1.5x and 6x.
	- Because it takes a self-supervised learning approach, V-JEPA is pre-trained entirely with unlabeled data. Labels are only used to adapt the model to a particular task after pre-training. This type of architecture proves more efficient than previous models, both in terms of the number of labeled examples needed and the total amount of effort put into learning even the unlabeled data. With V-JEPA, we’ve seen efficiency boosts on both of these fronts.
	- With V-JEPA, we mask out a large portion of a video so the model is only shown a little bit of the context. We then ask the predictor to fill in the blanks of what’s missing—not in terms of the actual pixels, but rather as a more abstract description in this representation space.
- [Google Gemma](https://huggingface.co/blog/gemma) - new open LLM
	- Gemma comes in two sizes: 7B parameters, for efficient deployment and development on consumer-size GPU and TPU and 2B versions for CPU and on-device applications. Both come in base and instruction-tuned variants.
	- bascally SOTA on Reasoning, Coding, [decent bump vs mistral on Math](https://www.reddit.com/r/LocalLLaMA/comments/1awbo84/google_publishes_open_source_2b_and_7b_model/), at the 2-7B range ([chart](https://x.com/Mascobot/status/1760365209720693150?s=20))
	- Gemma license has [odd terms](https://ai.google.dev/gemma/prohibited_use_policy)
	- [Karpathy breaks down Gemma tokenizer](https://x.com/karpathy/status/1760350892317098371?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): TLDR this is basically the Llama 2 tokenizer, except bigger (32K -> 256K), with a lot more special tokens, and the only functional departure is that add_dummy_prefix is turned off to False.
	- open source support - [llama.cpp, quantization, mlx, lmstudio](https://x.com/altryne/status/1760371315641397331?s=20)
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
- [AnimateLCM-SVD-xt](https://x.com/_akhaliq/status/1759658004880740765?s=20): AnimateLCM-SVD-xt can generally produces videos with good quality in 4 steps without requiring the classifier-free guidance, and therefore can save 25 x 2 / 4 = 12.5 times compuation resources compared with normal SVD models


## open source tooling and projects

- [Ollama openai compatibility APIs](https://news.ycombinator.com/item?id=39307330)
- [LlamaIndex v0.10](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8?source=collection_home---6------0-----------------------)
- [OpenLLMetry-JS](https://news.ycombinator.com/item?id=39371297) SDK via traceloop
	- an open protocol and SDK, based on OpenTelemetry, that provides traces and metrics for LLM JS/TS applications and can be connected to any of the 15+ tools that already support OpenTelemetry.
	- JS version of [Python one](https://news.ycombinator.com/item?id=37843907)
- [Reor – An AI note-taking app that runs models locally](https://github.com/reorproject/reor)

## product launches

- [Groq runs Mixtral 8x7B-32k with 500 T/s](https://news.ycombinator.com/item?id=39428880)
	- [groqchat](https://news.ycombinator.com/item?id=38739199)
	- [instant refactoring](https://twitter.com/mattshumer_/status/1759652937792422188)
	- [swyx recap ](https://twitter.com/swyx/status/1759759125314146699)
	- breakdown on [their marketing/comms](https://twitter.com/lulumeservey/status/1760401126287945830)
	- some talk aout [their compiler and scheduling tech](https://x.com/tomjaguarpaw/status/1759529106830479657?s=20)
- vercel ai integrations - aggregation theory on AI
- [Llamacloud and LlamaParse](https://news.ycombinator.com/item?id=39443972)
- [Lexica Aperture v4](https://x.com/sharifshameem/status/1760342835994439936?s=20) - near 4k resolution images. [more demos](https://x.com/sharifshameem/status/1760348586691408213?s=20)
- [Retell AI launch](https://news.ycombinator.com/item?id=39453402) - [Conversational Speech API for Your LLM](https://news.ycombinator.com/item?id=39453402) talked up by [Garry Tan and Aaron Levie](https://x.com/levie/status/1760415616157298816?s=20)
	- compare with [Vocode](https://github.com/vocodedev/vocode-python) and [Livekit](https://livekit.io/kitt)

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
- This month, the venture capital firm Menlo Ventures closed a deal to invest $750 million in Anthropic.
	- [NYT on Anthropic fundraising](https://news.ycombinator.com/item?id=39456140#39457725)
	- ~750m ARR 2024 projection - downshift from 850m in jan



## Misc reads

- discussions
	- [Umichigan selling student data](https://x.com/suchenzang/status/1758020313689006374?s=20) 
	- [Interesting discussion on Replicate's poor GPU cold start with CEO](https://news.ycombinator.com/item?id=39411748)
	- Gemini imagegen reverse racism issues
		- [gemini on australian, american, british, german women](https://x.com/debarghya_das/status/1759786243519615169?s=20)
		- [English women](https://x.com/RazRazcle/status/1760091322629898712?s=20), [medieval british kings](https://x.com/stratejake/status/1760333904857497650?s=46&t=90xQ8sGy63D2OtiaoGJuww)
		- [moralisticness on basic questions](https://x.com/RazRazcle/status/1760107387955126618?s=20)
		- [1930s germany](https://x.com/yacineMTB/status/1759971118457245965?s=20)
		- [google only acknowledged the problem with historical figures](https://twitter.com/benthompson/status/1760452419627233610?t=90xQ8sGy63D2OtiaoGJuww) but its bigger than that
		- similar but worse than [openai dalle](https://twitter.com/swyx/status/1760399621543879016) and [meta emu](https://x.com/andrewb10687674/status/1760414422990754071?s=20)
	- [Things I Don't Know About AI - Elad Gil](https://news.ycombinator.com/item?id=39453622)
- learning
	- **[TPU-Alignment](https://github.com/Locutusque/TPU-Alignment)** - Fully fine-tune large models like Mistral-7B, Llama-2-13B, or Qwen-14B completely for free. on the weekly 20hrs of TPUv3-8 pod from Kaggle 
	- [undo llama2 safety tuning with $200 LoRA](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?)
	- Karpathy [GPT Tokenizer](https://news.ycombinator.com/item?id=**39443965******) together with https://github.com/karpathy/minbpe
		- [practical application on vocab size vs tokens/s on codellama vs gpt3.5](1760477997994492272)

## memes

- https://twitter.com/JackPosobiec/status/1753416551066181672 dignifAI
- https://www.goody2.ai/chat
	- [the model card is also hilarious](https://www.goody2.ai/goody2-modelcard.pdf?utm_source=ainews&utm_medium=email)
- image of no elephant https://www.reddit.com/r/OpenAI/comments/1anm3p3/damn_sneaky/
	- source of meme: https://twitter.com/GaryMarcus/status/1755476468157833593
- apple vision pro https://discord.com/channels/822583790773862470/839660725252784149/1204047864096096328
- chatgpt memory x tipping meme https://x.com/_mira___mira_/status/1757695161671565315?s=46&t=90xQ8sGy63D2OtiaoGJuww
- karpathy leaving meme https://x.com/yacinemtb/status/1757775133761057113?s=46&t=90xQ8sGy63D2OtiaoGJuww
- gemini memes
	- this meme probably wins [the whole month](https://x.com/thecaptain_nemo/status/1760367238161604804?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- [everyone dogpiling on gemini](https://x.com/growing_daniel/status/1760459653887168984?s=20)
	- c.f. [James Damore](https://twitter.com/yacineMTB/status/1760460012848037919/photo/1). [paulg comment](https://x.com/paulg/status/1760416051181793361?s=20) and [carmack](https://x.com/ID_AA_Carmack/status/1760360183945965853?s=20)