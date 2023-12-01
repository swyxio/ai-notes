
- https://www.thurrott.com/cloud/290661/report-github-copilot-loses-an-average-of-20-per-user-per-month
- https://www.reuters.com/technology/chatgpt-owner-openai-is-exploring-making-its-own-ai-chips-sources-2023-10-06/
- https://blog.replit.com/replit-code-v1_5
- model transparency?
	- https://crfm.stanford.edu/fmti/ The Foundation Model Transparency Index
	- https://www.nytimes.com/2023/10/19/technology/allen-institute-open-source-ai.html
- state of ai report https://www.stateof.ai/
	- 1.  **GPT-4 is the master of all it surveys (for now),** beating every other LLM on both classic benchmarks and exams designed to evaluate humans, validating the power of proprietary architectures and reinforcement learning from human feedback.
	1.  **Efforts are growing to try to clone or surpass proprietary performance,** through smaller models, better datasets, and longer context. These could gain new urgency, amid concerns that human-generated data may only be able to sustain AI scaling trends for a few more years.
	    
	3.  **LLMs and diffusion models continue to drive real-world breakthroughs,** especially in the life sciences, with meaningful steps forward in both molecular biology and drug discovery.
	    
	4.  **Compute is the new oil,** with NVIDIA printing record earnings and startups wielding their GPUs as a competitive edge. As the US tightens its restrictions on trade restrictions on China and mobilizes its allies in the chip wars, NVIDIA, Intel, and AMD have started to sell export-control proof chips at scale.
	    
	5.  **GenAI saves the VC world, as amid a slump in tech valuations,** AI startups focused on generative AI applications (including video, text, and coding), raised over $18 billion from VC and corporate investors.
	    
	6.  **The safety debate has exploded into the mainstream,** prompting action from governments and regulators around the world. However, this flurry of activity conceals profound divisions within the AI community and a lack of concrete progress towards global governance, as governments around the world pursue conflicting approaches.
	    
	7.  **Challenges mount in evaluating state of the art models,** as standard LLMs often struggle with robustness. Considering the stakes, as “vibes-based” approach isn’t good enough.


## openai

- Exciting news for @OpenAI devs: we are close to a 1.0 release of the OpenAI Python SDK 🎊. You can test the beta version of 1.0 today, we would love to get your early feedback! [changelog](https://twitter.com/StainlessAPI/status/1709650331972379060)
- sama podcast on joe rogan experience
- finetuning ui has end to end job creation, no code required ([tweet](https://twitter.com/OfficialLoganK/status/1710434050274734110) - [not our fault](https://x.com/OfficialLoganK/status/1710680533574045781?s=20))
- [openai dalle3 in chatgpt plus and enterprise](https://openai.com/blog/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise) together with [research paper](https://cdn.openai.com/papers/dall-e-3.pdf)
- [OpenAI’s technology explained](https://twitter.com/OfficialLoganK/status/1712483165380415828)
- Collection of [ChatGPT System Prompts](https://news.ycombinator.com/item?id=37879077) including Voice and Custom Instructions
- gpt4 date updated to apr 2023 https://x.com/simonw/status/1717626503121576435?s=20
- gpt4 all tools
	- https://twitter.com/DataChaz/status/1719660354743976342 32k context
- preparedness challenge https://news.ycombinator.com/item?id=38029307
- openai new office
- openai 86b tender offer https://web.archive.org/web/20231027165729mp_/https://www.afr.com/markets/equity-markets/openai-is-in-talks-to-sell-shares-at-136b-valuation-20231020-p5edqj


## other frontier models
- -   Inflection Pi got internet access & New therapy mode ([Announcement](https://substack.com/redirect/f141031f-3686-47d7-a62c-897237629219?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE))
- -   Anthropic [Claude.ai](http://claude.ai/) is rolled out to additional 95 countries ([Announcement](https://substack.com/redirect/077443fe-35d4-4f0d-aa59-88524f729880?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE))
- all Metamates [given GPT-4 access internally](https://x.com/JBasedos/status/1719381484413571510?s=20), while Google Gemini is still nowhere to be found
    

## Models

- Large
  - Lemur-70B & Lemur-70B-Chat: 🚀Open & SOTA Foundation Models for Language Agents! The closest open model to GPT-3.5 on 15 agent tasks ([tweet](https://twitter.com/yihengxu_/status/1712537543688990940), [paper](https://arxiv.org/abs/2310.06830))
- Medium
	- [Mistral 7B paper](https://arxiv.org/pdf/2310.06825.pdf) 
		- [mission](https://x.com/abacaj/status/1709455939231772962?s=20) "Our ambition is to become the leading supporter of the open generative Al community, and bring open models to state-of-the-art
			performance. We will make them the go-to solutions for most of the generative Al applications. Many of us played pivotal roles in
			important episodes in the development of LLMs; we're thrilled to be working together on new frontier models with a community-
			oriented mindset.
			In the coming months, Mistral Al will progressively and methodically release new models that close the performance gap between
			black-box and open solutions - making open solutions the best options on a growing range of enterprise use-cases.
			Simultaneously, we will seek to empower community efforts to improve the next generations of models."
		- Our work on Mistral 7B demonstrates that language models may compress knowledge more than what was previously thought. This opens up interesting perspectives: the field has so far put the emphasis on scaling laws in 2 dimensions (directly associating model capabilities to training cost) the problem is rather 3 dimensional (model capabilities, training cost, inference cost), and much remains to be explored to obtain the best performance with the smallest possible model.
		- Mistral 7B outperforms Llama 13B on all tested benchmarks
		- introduces [Sliding Window Attention](https://twitter.com/marktenenholtz/status/1707412664308138218) 
			- Pieter Abbeel: "[convnets again?](https://x.com/AravSrinivas/status/1712337087306092914?s=20)"
			- likely a [red herring](https://x.com/main_horse/status/1712340876633235526?s=20)
			- [no info on data at all](https://x.com/leavittron/status/1715247384941498761?s=20)
	- zephyr 7b
		- Zephyr 7b beats Llama 70b on MT Bench https://twitter.com/huggingface/status/1711780979574976661
		- [major training steps](https://twitter.com/thom_wolf/status/1717821614467739796?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- [only LLM that can handle ReAct agent tasks over data](https://twitter.com/llama_index/status/1718054631413363196)
		- [nathan lambert take](https://twitter.com/natolambert/status/1711767611586035884?s=12&t=90xQ8sGy63D2OtiaoGJuww) 
		- noncommercial license [for now](https://x.com/natolambert/status/1711947441057276227?s=20)
	- Announcing Open Hermes 2!! A continuation of the Hermes series of models, now. built on Mistral 7B! The Hermes 2 model was trained on 900,000 instructions, and surpasses all previous versions of Hermes 13B and below, and matches 70B on some benchmarks! Hermes 2 changes the game with strong multiturn chat skills, system prompt capabilities, and uses ChatML format. It's quality, diversity and scale is unmatched in the current OS LM landscape. Not only does it do well in benchmarks, but also in unmeasured capabilities, like Roleplaying, Tasks, and more. [https://fxtwitter.com/Teknium1/status/1714010838959612329](https://fxtwitter.com/Teknium1/status/1714010838959612329 "https://fxtwitter.com/Teknium1/status/1714010838959612329")
- Small
  - [ Stable LM 3B: Bringing Sustainable, High-Performance LMs to Smart Devices](https://stability.ai/blog/stable-lm-3b-sustainable-high-performance-language-models-smart-devices) https://news.ycombinator.com/item?id=37739965
  - Rift coder 7B https://twitter.com/morph_labs/status/1709288051195998375 finetuned from glaive-coder-7b.
  - [CollectiveCognition v1](https://twitter.com/Teknium1/status/1709750388528939473) finetune of Mistral 7B
  - [TinyLanguage Models come of age](https://news.ycombinator.com/item?id=37787350)
  - Adept multimodal model ([twitter]([https://twitter.com/AdeptAILabs/status/1714682075763405257](https://twitter.com/AdeptAILabs/status/1714682075763405257 "https://twitter.com/AdeptAILabs/status/1714682075763405257"))) - Fuyu-8B
	  - compare to nougat  https://facebookresearch.github.io/nougat/:  an open-source OCR model that accurately scans books with heavy math/scientific notations. It's ages ahead of other open OCR options. Meta is doing extraordinary open-source AI, sometimes without as much fanfare as Llama. ([twitter](https://twitter.com/DrJimFan/status/1702322181928259586))
  - [gte-tiny](https://x.com/xenovacom/status/1710347897793810586?s=20) - a distilled version of gte-small that is half the size (20m params). [bge-micro is 5m params](https://twitter.com/andersonbcdefg/status/1710708732534108413)

- ?
  - Reka Yasa-1 Multimodal LLM ([tweet](https://twitter.com/YiTayML/status/1709265184576204820), [blog](https://reka.ai/announcing-our-multimodal-ai-assistant/) - natively supports images, audio, and short video clips as inputs. )
    - Retrieval augmented generation - Yasa can be taught to understand private datasets. Our API and on-premise deployment setup allows seamless integration of internal datasets of any modality type.
    - Code Interpreter - Yasa is more than just a passive AI assistant; it has the capability to actively execute code. This feature is enabled via a simple flag. When active, Yasa automatically identifies the code block within its response, executes the code, and appends the result at the end of the block.

## open source projects and templates

- Daniel Gross’ [LocalPilot](https://x.com/danielgross/status/1708855228122964291?s=20)- “In my experience, 7b isn't usefully fast enough for autocomplete on M1, but M2 Max is the punctuated equilibrium; it's suddenly good enough. (34b quantized models are fast enough for Q&A.)“
- [headshot AI project](https://x.com/svpino/status/1711003548073504886?s=20)
  - uses https://twitter.com/leap_api
- Llama 2 in C ([Karpathy]( https://x.com/karpathy/status/1710061549677613469?s=46&t=6FDPaNxZcbSsELal6Sv7Ug))
  - [Llama 2 Everywhere (L2E): Standalone, Binary Portable, Bootable Llama 2](https://github.com/trholding/llama2.c)https://news.ycombinator.com/item?id=37785442
- Local LLM calculator: [select LLM, and GPU, and see if can run locally](https://x.com/victormustar/status/1712754193784520966?s=20)
- [SlowLlama: Finetune llama2-70B and codellama on MacBook Air without quantization](https://github.com/okuvshynov/slowllama)
- Cloudflare AI projects
	- [LlamaResearcher](https://twitter.com/ritakozlov_/status/1712100386717237603?s=12&t=90xQ8sGy63D2OtiaoGJuww): personalized research assistant that sends summaries of papers of interest straight to your inbox
- Finetunes
	- https://dadjokes.dfdx.me/ mistral finetune on /r/dadjokes
- fast whisper distributions
	- [whisper turbo](https://whisper-turbo.com) - purely in browser ([tweet context](https://twitter.com/fleetwood___/status/1709364288358662479)), using webgpu
	- [ Distil-Whisper: distilled version of Whisper that is 6 times faster, 49% smaller](https://github.com/huggingface/distil-whisper)
- Redpajama Dataset v2 https://news.ycombinator.com/item?id=38077521

## other launches

- Stanformd CRFM [ecosystem graphs](https://crfm.stanford.edu/ecosystem-graphs/index.html?mode=table): tracks foundation models, upstream datasets, and downstream products
- Langchain langserve - https://twitter.com/LangChainAI/status/1712526285313102091
	- [Langchain templates](https://twitter.com/langchainai/status/1719377131313172556) - a collection of easily deployable reference architectures for a wide variety of tasks (aka [Langserve Hub]([https://blog.langchain.dev/langserve-hub/](https://t.co/8YO5pjlPWl)))
- Lexica Aperture 3.5
- Pplx Mistral 7B, Llama2 13B, Code Llama 34B, and Llama2 70B models supported [https://blog.perplexity.ai/blog/introducing-pplx-api](https://blog.perplexity.ai/blog/introducing-pplx-api "https://blog.perplexity.ai/blog/introducing-pplx-api")
  - currently included with perplexity pro, no $/token (for now? I'm assuming only in public beta, that won't scale)
  - really leans into openAI api compatibility. They actually just use the openai python client. All you have to do is switch api_base, api_key, and model to point to perplexity to switch an application from openai to perplexity in python
  - (thanks @danjl on Latent Space Discord)
- [ Show HN: Shortbread – Create AI comics in minutes](https://shortbread.ai/) ([shortbread.ai](https://news.ycombinator.com/from?site=shortbread.ai))
  - great demo of what a nice stable diffusion 1.5 app can do today
- [SiFive Rolls Out RISC-V Cores Aimed at Generative AI and ML](https://www.allaboutcircuits.com/news/sifive-rolls-out-risc-v-cores-aimed-at-generative-ai-and-ml/) https://news.ycombinator.com/item?id=37911544
  - problems with sifive - article
- [Show HN: Riffusion with Lyrics](https://www.riffusion.com/) ([riffusion.com](https://news.ycombinator.com/from?site=riffusion.com))
	- https://news.ycombinator.com/item?id=37914425
	- now generating short song segments with VOICE!
- [Show HN: OpenLLMetry – OpenTelemetry-based observability for LLMs](https://github.com/traceloop/openllmetry) ([github.com/traceloop](https://news.ycombinator.com/from?site=github.com/traceloop)) https://news.ycombinator.com/item?id=37843907
- Midjourney upscalers
- **[3.](https://link.mail.beehiiv.com/ss/c/6TtoGq78XEbq5fhDZPJ0UVM7mdUOzNgajUjP1A--As3xx7u0DdLZd6dAVfjB8wC7LLYbjJBHTtx7gmR5dll5FYWFI45_fdxHBgFBkVPVZ8Y5RHybKh3pX9ZeBvcsUA2sp2JEHOms2n5wt10a1HGWI9NCVMT6cnD2_SddJRc7eZfkg3FtnsgiQD1R99qMeq3VsfR21aOFj3kyxDMLfa91fA/40j/Q17mfsZ4Srqxxs9FaTUGNA/h6/CoyioJ8d8vP0gRHB2kwwQ7DKl9gHzzDRSN2TmLWoDaw)** **[Adobe releases their Firefly 2 Image Model](https://link.mail.beehiiv.com/ss/c/6TtoGq78XEbq5fhDZPJ0UVM7mdUOzNgajUjP1A--As3xx7u0DdLZd6dAVfjB8wC7LLYbjJBHTtx7gmR5dll5FYWFI45_fdxHBgFBkVPVZ8Y5RHybKh3pX9ZeBvcsUA2sp2JEHOms2n5wt10a1HGWI9NCVMT6cnD2_SddJRc7eZfkg3FtnsgiQD1R99qMeq3VsfR21aOFj3kyxDMLfa91fA/40j/Q17mfsZ4Srqxxs9FaTUGNA/h7/TOp8BBRxsZX651R0366F7cKtozT2tNfN-cvdA5Nj-l0)**  Adobe Firefly Image 2 offers enhanced architecture and training algorithms for high-quality, photorealistic image generation. It supports 4MP output, depth of field control, and high-frequency details like skin pores. Features like Generative Match enable style transfer from reference images.
- Baidu claims their new model Ernie 4 rivals GPT4 but cant be proven - ERNIE 4, multimodal foundational model, integrated with many applications ([Announcement](https://substack.com/redirect/163df6af-340d-4957-88c3-d911531f7284?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE), [Thread](https://substack.com/redirect/c3d12d8a-4560-46ed-87e4-bdb95ae5e01e?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE))
- Adobe releases Firefly 2 - lifelike and realistic images, generative match, prompt remix and prompt suggestions ([X](https://substack.com/redirect/de20b453-ad97-499e-b02f-995d460477c8?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE), Firefly)
- [Play.ht](http://play.ht/) shows off an impressive <300ms voice generation for agents After spending almost 2 hours talking to chatGPT, I was thinking, why aren't all AI assistants like this, and the answer was, well... generating voice takes time, which takes you out of your "conversation flow" And then today, [play.ht](https://substack.com/redirect/fe90d888-07d3-4441-9301-3c8bdbe35c0a?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE) showed off a new update to their API that generates voice in <300ms, and that can be a clone of your voice, with your accent and all. We truly live in unprecedented times.
- [Defog Agents: AI Assistants for complex data workflows](https://defog.ai/blog/agents/)
- Databricks [MLflow 2.8](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part?utm_source=twitter&utm_medium=organic-social) supports LLM-as-a-judge metrics - resulting in significant savings in time (from 2 weeks with human workforce to 30 minutes with LLM judges) and costs (from $20 per task to $0.20 per task)
- [Morph Code Index](https://x.com/morph_labs/status/1711803918764966017?s=20) is an OSS semantic code search engine for you, your codebase, and your personal AI SWE.

## Papers and Good Reads

- Nathan Benaich's annual [State of AI Report](https://twitter.com/nathanbenaich/status/1712358033194688701?s=12&t=90xQ8sGy63D2OtiaoGJuww) is out
- Models
	- [Efficient streaming language models with attention sinks](https://github.com/mit-han-lab/streaming-llm) ([HN](https://news.ycombinator.com/item?id=37740932#37742452))
		-   **Attention Sinks:** Use and maintain "attention sinks", initial tokens that the model focuses on.
		-   **Rolling Cache:** Keep a rolling collection of recent tokens to optimize speed without sacrificing accuracy.
		-   **Placeholder Token:** Add a special token during training to act as a dedicated attention sink, enhancing streaming deployment.
	- [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226) ([HN](https://news.ycombinator.com/item?id=37764382))
		- adding up to 10 "pause tokens" lets models improve reasoning - tested up to 1B params on C4
		- seems similar to [the backspace token paper](https://arxiv.org/pdf/2306.05426)
	- [Expressive text-to-image generation with rich text](https://rich-text-to-image.github.io/) ([HN](https://news.ycombinator.com/item?id=37770260)) - being able to modify generated images by modifying text using fonts and text colors. going from words -> token maps (masks).
	- [Notable criticism of this method over regular prompting](https://news.ycombinator.com/item?id=37772250)
	- [FontoGen](https://serce.me/posts/02-10-2023-hey-computer-make-me-a-font) -  The model takes a font description as an input, and produces a font file as an output.
	- NEFTune - a "one simple trick" to get higher quality finetunes by adding noise ([Thread](https://substack.com/redirect/ded46fe9-fe20-4547-8166-696390a6a05b?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE), [Github](https://substack.com/redirect/434db4cd-f376-4aef-81ef-0d6c4aa34f0d?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE))
- Prompting
  - emotional jailbreaks work. [https://arxiv.org/abs/2307.11760](https://arxiv.org/abs/2307.11760) "This is very important to my career" and [the dead grandma jailbreak](https://news.ycombinator.com/item?id=37743759)
  - [https://arxiv.org/abs/2310.04406v1](https://arxiv.org/abs/2310.04406v1 "https://arxiv.org/abs/2310.04406v1") 94.4 on HumanEval with gpt4 86.9 on HumanEval with gpt3.5 wild
  - [Large Language Models as Analogical Reasoners](https://arxiv.org/pdf/2310.01714.pdf) from Deepmind
  - -   When given a task, the LLM is prompted to:
      -   First, create relevant examples (problems and their solutions) for the task.  
      -   Then, use these examples as guidance to solve the main task.
  - [using Midjourney and GPT4 to code an Angry Birds clone](https://twitter.com/javilopen/status/1719363262179938401)
- RAG
  - https://arxiv.org/pdf/2310.05029
  - RAG vs Long Context
    - https://x.com/_philschmid/status/1710575136657600958?s=46&t=6FDPaNxZcbSsELal6Sv7Ug
    - https://arxiv.org/abs/2310.03025
  - [Vector database is not a separate database category](https://nextword.substack.com/p/vector-database-is-not-a-separate) ([HN](https://news.ycombinator.com/item?id=37747534))
  - Text embeddings can be inverted ([twitter](https://x.com/jxmnop/status/1712562908133999069?s=20), [paper](https://arxiv.org/pdf/2310.06816.pdf))
	  - "These results imply that text embeddings present the same threats to privacy as the text from which they are computed, and embeddings should be treated with the same precautions as raw data."
	  - "Vec2Text is trained to invert two state-of-the-art embedding models: GTR-base (Niet al., 2021), a T5-based pre-trained transformer for text retrieval, and text-embeddings-ada-002 available via the OpenAI API"
	  - "Vec2Text is able to recover 94% of first names, 95% of last names, and 89% of full names (first, last format) while recovering 26% of the documents exactly."
  - MemGPT - LLMs as operating systems ([twitter](https://x.com/iScienceLuvr/status/1712747095676117155?s=20), [site](https://memgpt.ai/), arxiv, [HN](https://news.ycombinator.com/item?id=37894403))
  - MemWalker - Long context models are popular, but is it the final solution to long text reading? We introduce a fundamentally different method, MemWalker: 1. Build a data structure (memory tree) 2. Traverse it via LLM prompting Outperforms long context, retrieval, & recurrent baselines. (1/n) ([tweet](https://twitter.com/__howardchen/status/1711584916708938042))
  - [ Choosing vector database: a side-by-side comparison](https://benchmark.vectorview.ai/vectordbs.html) ([HN](https://news.ycombinator.com/item?id=37764489))
    - "Everyone I talk to who is building some vector db based thing sooner or later realizes they also care about the features of a full-text search engine.
    - They care about filtering, they care to some degree about direct lexical matches, they care about paging, getting groups / facet counts, etc.
    - Vectors, IMO, are just one feature that a regular search engine should have. IMO currently Vespa does the best job of this, though lately it seems Lucene (Elasticsearch and Opensearch) are really working hard to compete"
  - [Vespa.ai is spinning out of Yahoo as a separate company](https://blog.vespa.ai/vespa-is-becoming-its-own-company/) people speak highly of Vespa, it is targeting search and recsys problems rather than specifically vector db problems
- Evals
  - [Evaluating LLMs is a minefield](https://twitter.com/random_walker/status/1709583031001124889)
    - the reports of ChatGPT having a liberal bias were a result of oversensitive prompts
    - GPT4 passing the bar exam and USMLE is a sign of contamination
      - "[OpenAI’s method to detect contamination is superficial and sloppy](https://www.aisnakeoil.com/p/gpt-4-and-professional-benchmarks)"
- Efficiency
  - [Running Stable Diffusion XL 1.0 in 298MB of RAM](https://github.com/vitoplantamura/OnnxStream/tree/846da873570a737b49154e8f835704264864b0fe)
    - OnnxStream is based on the idea of decoupling the inference engine from the component responsible of providing the model weights, which is a class derived from `WeightsProvider`. A `WeightsProvider` specialization can implement any type of loading, caching and prefetching of the model parameters. For example a custom `WeightsProvider` can decide to download its data from an HTTP server directly, without loading or writing anything to disk (hence the word "Stream" in "OnnxStream"). Two default `WeightsProviders` are available: `DiskNoCache` and `DiskPrefetch`.
  - Mistral 7B finetuning guide
	  - guide showing how to fine-tune it cost-effectively using QLoRA ([brev tweet](https://twitter.com/HarperSCarroll/status/1709000201963532429))
	  - guide from wandb: https://news.ycombinator.com/item?id=37813806
  - [LLMs overview from Flyte](https://flyte.org/blog/getting-started-with-large-language-models-key-things-to-know#what-are-llms)
  - [OpenAI is too cheap to beat](https://generatingconversation.substack.com/p/openai-is-too-cheap-to-beat) - "At $366K ($166K AWS + $200K talent), we’re paying around $80 per-fine-tuning run, about 8-20x higher than what we’re paying OpenAI!"
    - [Open, general-purpose LLM companies might not be viable](https://www.interconnects.ai/p/are-open-llms-viable)
- Agents
	- [OpenAgents: An Open Platform for Language Agents in the Wild](https://arxiv.org/abs/2310.10634)
		- open replication off chatgpt plus tools based on their own comparison table (see [chart](https://twitter.com/itsuka_dev/status/1715516618791702783/photo/2)) using 1. Data agent: data analysis and tools, 2. Plugin agent: 200+ APIs from API provider: RapidAPI, OAI plugin store, 3. Web agent (Tools: Chrome debugger API)
		- from the Executable Language Grounding (XLang) Lab! We are part of the HKU NLP Group at the University of Hong Kong.
		- didnt find a lot of insight while reading thru. too much marketing in the paper
- Multimodality
	- LMMs > LLMs
	- [Multi-modal prompt injection image attacks against GPT-4V](https://simonwillison.net/2023/Oct/14/multi-modal-prompt-injection/) ([simonwillison.net](https://news.ycombinator.com/from?site=simonwillison.net))
	- [Multimodality and Large Multimodal Models (LMMs)](https://huyenchip.com//2023/10/10/multimodal.html)
	- [Ferret: Refer and Ground Anything Anywhere at Any Granularity](https://arxiv.org/abs/2310.07704) - nice attempt at Open GPT4-V, and has a nice GRIT dataset others can use
	- meta released MetaCLIP - fully OSS replication of CLIP pipeline
		- Paper: [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671)
- [The Killer Use Case for LLMs Is Summarization](https://www.sebastianmellen.com/post/2023/the-killer-use-case-for-llms-is-summarization/) ([sebastianmellen.com](https://news.ycombinator.com/from?site=sebastianmellen.com))
	- https://news.ycombinator.com/item?id=37946023
- learning
	- This page contains interactive charts for exploring how large language models represent truth https://saprmarks.github.io/geometry-of-truth/dataexplorer/
	- [Language Modeling is Compression](https://arxiv.org/pdf/2309.10668) from DeepMind echoes what Ilya Sutskever said in a [recent talk](https://youtu.be/GI4Tpi48DlA?si=u7guXWVvIS5OsRaH)
	- Large Language Models in 2023 ([tweet](https://twitter.com/hwchung27/status/1710003293223821658?s=12&t=90xQ8sGy63D2OtiaoGJuww), [recorded talk](https://www.youtube.com/watch?app=desktop&v=dbo3kNKPaUA&feature=youtu.be)) from Hyung Won Chung, OpenAI & Google Brain
		- emergence (Wei et al) is still underappreciated
		- Perspective of "yet": "This idea doesn't work" -> "This idea doesn't work YET" 
			- Document experiments that failed because of insufficient “intelligence”
			- Do not declare failure yet and make it easy to rerun in the future
			- As soon as the new model comes out, rerun them
			- Learn what works and what doesn’t
			- Update your intuition on emergent abilities and scale
		- We need post-training
			- instruction tuning - FLAN
			- Reward Model training
			- Policy model training
		- bitter lesson
			- Many Transformer variants have been proposed but almost all fancy variations don’t scale well
			- More useful to abstract away Transformer as sequence of functions and think about input and output shapes and types
- misc
	- [PaLI-3 Vision Language Models](https://arxiv.org/abs/2310.09199) ([arxiv.org](https://news.ycombinator.com/from?site=arxiv.org))
	- TimeGPT https://news.ycombinator.com/item?id=37874891 

## Fundraising

- Modal Labs Series A
  - https://x.com/modal_labs/status/1711748224610943163?s=20
- Anyscale (Cursor.so) Seed
  - https://techcrunch.com/2023/10/11/anysphere-raises-8m-from-openai-to-build-an-ai-powered-ide/
- Induced AI $2.3m seed
  - We let anyone create virtual AI workers that can automate the execution of workflows on a browser in the cloud with human-like reasoning.
  - https://twitter.com/aryxnsharma/status/1709289742310010970
- Anthropic funding
	- google
	- amazon

## misc & prior discussions

- watermarking
  - researchers broke all watermarks? https://news.ycombinator.com/item?id=37767633
  - Truepic C2PA content credentials + Huggingface https://twitter.com/mmitchell_ai/status/1710123404706378233
- Custom Instructions - nice template https://github.com/spdustin/ChatGPT-AutoExpert/
- [Security weaknesses of Copilot generated code in GitHub](https://arxiv.org/abs/2310.02059)
	- "If a weakness is common, then of course Copilot is going to suggest it. Copilot gives you popular responses not correct ones. Yet if a weakness is common, it also means that human coders frequently make the same mistake as well."
- Phind defaults to their own model that beats GPT4 at coding with GPT3.5 speed ([HN](https://news.ycombinator.com/item?id=38088538), previously reported)
- Pmarca [Techno-Optimist Manifesto](https://a16z.com/the-techno-optimist-manifesto/) 
	- [Marc Andreessen's AI manifesto hurts his own cause](https://www.axios.com/2023/10/17/marc-andreessens-ai-manifesto-hurts-his-own-cause) ([axios.com](https://news.ycombinator.com/from?site=axios.com))
- Mojo 🔥 is now available on Apple silicon Macs and has LLaMa.cpp level performance ([Announcement](https://news.ycombinator.com/item?id=37942574), [Performance thread](https://substack.com/redirect/57b061be-a190-4942-8b81-564e85d77749?j=eyJ1IjoiMmpqYnkxIn0.voZ98EfOPBt5Ku2V2Dg7KBxwdLf5SmXhj0TZ3U45rJE))
- safety
	- Biden executive order
	- [Andrew Ng](https://web.archive.org/web/20231027165729mp_/https://www.afr.com/markets/equity-markets/openai-is-in-talks-to-sell-shares-at-136b-valuation-20231020-p5edqj)  tweet
- is copilot making money?
	- losing 20-80/month/user https://twitter.com/abacaj/status/1711754522794533046?s=12&t=90xQ8sGy63D2OtiaoGJuww
	- but denied and 100m ARR
- [The New Yorker on the future of training methods](https://www.newyorker.com/science/annals-of-artificial-intelligence/how-will-ai-learn-next) ([HN](https://news.ycombinator.com/item?id=37785416))

## memes

- grandma is the new sudo https://twitter.com/latentspacepod/status/1708982643294146690
- its a rock https://x.com/itsmingjie/status/1709039235913719973?s=46&t=90xQ8sGy63D2OtiaoGJuww
- mistral gigachad memes https://twitter.com/nearcyan/status/1714850537194025208?s=12&t=90xQ8sGy63D2OtiaoGJuww
- aggressive apple eating was a meme? https://twitter.com/abacaj/status/1715034087902032253?s=12&t=90xQ8sGy63D2OtiaoGJuww