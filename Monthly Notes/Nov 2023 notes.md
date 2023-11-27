
## monthly theme

- voice cloning red alert
	- voice cloning https://twitter.com/martin_casado/status/1726707806492131819
	- and here https://x.com/notcapnamerica/status/1725901659270742084?s=20
	- https://github.com/w-okada/voice-changer/
		- https://www.weights.gg/
- consistency models and GPT4V coding
	- Krea
	- TLDraw
- people need to calm the fuck down 
	- Q* mass hysteria 
		- with some 4chan leaks? https://docs.google.com/document/u/0/d/1RyVP2i9wlQkpotvMXWJES7ATKXjUTIwW2ASVxApDAsA/mobilebasic?utm_source=ainews&utm_medium=email&utm_campaign=ainews-ai-discords-newsletter-11272023
	- OpenAI leadership drama


## OpenAI
- [Ilya Sutskever on No Priors pod](https://www.youtube.com/watch?v=Ft0gTO2K85A)
- dev day
	- GPTs: Custom versions of ChatGPT - https://news.ycombinator.com/item?id=38166431
	- OpenAI releases Whisper v3, new generation open source ASR model - https://news.ycombinator.com/item?id=38166965
		- https://twitter.com/YoungPhlo_/status/1721606700082336013
	- OpenAI DevDay, Opening Keynote Livestream [video] - https://news.ycombinator.com/item?id=38165090
	- vision
		- openinterpreter integration https://twitter.com/hellokillian/status/1723106008061587651
	- gpts
		- [gdb personal favorite](https://twitter.com/gdb/status/1721638249414947235)
		- useful for low level systems developers [reading docs](https://news.ycombinator.com/item?id=38192094)
		- https://news.ycombinator.com/item?id=38166431
		- typist https://twitter.com/_Borriss_/status/1725208054185046071
		- prompts for [many many other GPTs](https://github.com/linexjlin/GPTs)
	- chatgpt memory "rumor" https://twitter.com/yupiop12/status/1724915477900656857
	- json mode
		- [does not guarantee output will match your schema](https://community.openai.com/t/json-mode-vs-function-calling/476994/6) but will improve
		- [functions param deprecated!! no notice!](https://x.com/Tangpin78255362/status/1722049314103705650?s=20)
	- dalle3
		- the consistency model VAE decoder (used in DALL-E 3) for the Stable Diffusion 1.4/1.5 VAE https://twitter.com/iScienceLuvr/status/1721601915975528683
- nov 14: paused chatgpt plus signups, ebay at premium https://x.com/sdand/status/1724629169483719104?s=20
- sama fired nov 17
	- was at cabridge https://www.youtube.com/watch?v=NjpNG0CJRMM
	- was at apec summit https://www.youtube.com/watch?v=ZFFvqRemDv8
	- you all know what happened
	- [sama's last interview](https://podcasts.apple.com/us/podcast/mayhem-at-openai-our-interview-with-sam-altman/id1528594034?i=1000635486878) before being fired
- gpt3.5 issues https://twitter.com/sharifshameem/status/1725422821730636236
	- zephyr great
- [interesting, nontechnical real life use cases for GPT4 from Reddit](https://www.reddit.com/r/OpenAI/comments/182zkdl/whats_the_hardest_real_life_problem_you_have/)

## models

- Claude 2.1
	- https://www.anthropic.com/index/claude-2-1
	- model card https://www-files.anthropic.com/production/images/ModelCardClaude2_with_appendix.pdf?dm=1700589594
	- $1k test showing [declining utilization](https://twitter.com/GregKamradt/status/1727018183608193393) of the 200k context
- Inflection 2
	- https://news.ycombinator.com/item?id=38380377
	- [5,000 NVIDIA H100 GPUs in fp8 mixed precision for ~10²⁵ FLOPs](https://x.com/mustafasuleyman/status/1727349979440984371?s=20). guess is [300b model](https://twitter.com/EMostaque/status/1727373950685200674) on 5T tokens
	- [slightly better than llama-2](https://x.com/Teknium1/status/1727353306132185181?s=20)
- [Yarn-Mistral-7b-128k](https://x.com/mattshumer_/status/1720115354884514042?s=20): 
	- 4x more context than GPT-4. Open-source is the new long-context king! This thing can easily fit entire books in a prompt.
	- [tweet](https://x.com/theemozilla/status/1720107186850877662?s=20)
- Yi 01 ai https://twitter.com/kaifulee/status/1721321096727994590
- Amazon Mistral - 32k
	- https://twitter.com/MatthewBerman/status/1719758392510824505
- Elon X.ai Grok
	- Grok-1 is a autoregressive Transformer-based model pre-trained to perform next-token prediction. The model was then fine-tuned using extensive feedbacks from both human and the early Grok-0 models. The initial Grok-1 has a context length of 8,192 tokens and is released in Oct 2023. https://twitter.com/altryne/status/1721028995456647286
	- grok vs openchat - https://twitter.com/alignment_lab/status/1721308271946965452
- Microsoft Phi-2 at Ignite - 2.7b params, 50% better at math
	- https://x.com/abacaj/status/1724850925741768767?s=20
	- 213b tokens
	- synthetic data from GPT3.5
- [Gorilla OpenFunctions](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html)
	- open-source function calling model, and are thrilled to present Gorilla OpenFunctions. And yes, we've made parallel functions a reality in open-source!
	- https://twitter.com/shishirpatil_/status/1725251917394444452
- Stable Video Diffusion
	- https://news.ycombinator.com/item?id=38368287
	- and video LLaVA https://news.ycombinator.com/item?id=38366830


### Fundraising

- Mistral @ 2bn https://archive.ph/hkWD3
- Factory.ai $5m seed https://www.factory.ai/blog https://x.com/matanSF/status/1720106297096593672?s=20
- [Modal labs raised $16m series A with Redpoint](https://twitter.com/modal_labs/status/1711748224610943163?s=12&t=90xQ8sGy63D2OtiaoGJuww)


## open source projects

- david attenborough project
	- launch https://news.ycombinator.com/item?id=38281079 
	- [github](https://github.com/cbh123/narrator)
	- [arstechnica](https://news.ycombinator.com/item?id=38302319)
- anyscale llm benchmarking standards https://twitter.com/robertnishihara/status/1719760646211010997
- Replicate labs
	- YouTune - [finetune image models on yt videos](https://x.com/charliebholtz/status/1719847667495231700?s=20)
- Finetuning
	- [gpt evaluator as judge](https://x.com/llama_index/status/1719868813318271242?s=20) - from llamaindex
	- [Learnings from fine-tuning LLM on my Telegram messages](https://github.com/furiousteabag/doppelganger)
- llama index + pplx api https://x.com/llama_index/status/1725975990911086932?s=46&t=90xQ8sGy63D2OtiaoGJuww
- sharing screen with gpt4 vision
	- https://twitter.com/suneel_matham/status/1722538037551530069
	- https://github.com/hackgoofer/PeekPal
- insanely-fast-whisper-with-diarisation https://twitter.com/reach_vb/status/1729251580371689821
- ggml - [tutorial for setting up llama.cpp on AWS instances](https://twitter.com/ggerganov/status/1729232091370369160)
	- you can use one of the cheapest 16GB VRAM (NVIDIA T4) instances to serve a quantum Mistral 7B model to multiple clients in parallel with full context.
	- 3B LLaMA provisioned for 1 month is $15k? You can run quantum 13B on the g4dn.xlarge instance above for ~$500/month

## other launches

- Elon Musk's X.ai Grok model was announced but not widely released [HN](https://news.ycombinator.com/item?id=38148396)
	- "A unique and fundamental advantage of Grok is that it has real-time knowledge of the world via the X platform."
	- The engine powering Grok is Grok-1, our frontier LLM, which we developed over the last four months. Grok-1 has gone through many iterations over this span of time. [Model Card](https://x.ai/model-card/)
	- [https://x.ai/](https://x.ai/) - Link to waitlist: [https://grok.x.ai/](https://grok.x.ai/) 
	- [UI demo](https://twitter.com/TobyPhln/status/1721053802235621734)
	- xAI PromptIDE is an integrated development environment for prompt engineering and interpretability research. https://x.ai/prompt-ide/ https://twitter.com/xai/status/1721568361883279850
- GitHub Copilot GA
	- https://twitter.com/HamelHusain/status/1723047256180355386
	- workspace https://twitter.com/ashtom/status/1722631796482085227
	- chat https://x.com/ashtom/status/1722330209867993471?s=20
	- mitchell: https://x.com/mitchellh/status/1722346134130348504?s=20
- Microsoft Ignite - copilot everything
	- https://www.theverge.com/23961007/microsoft-ignite-2023-news-ai-announcements-copilot-windows-azure-office
- Adept Experiments: https://www.adept.ai/blog/experiments
	- Workflows: Workflows is powered by ACT-2, a model fine-tuned from the Fuyu family and optimized for UI understanding, knowledge worker data comprehension, and action taking. 
- consistency models
	- krea https://news.ycombinator.com/item?id=38223822
	- live hf space https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model
	- latent consistency and LCM-LoRA are the incremental innovations that spawned the recent work https://huggingface.co/collections/latent-consistency/latent-consistency-model-demos-654e90c52adb0688a0acbe6f
		- https://twitter.com/javilopen/status/1724398666889224590
- TLDraw [Make real](https://x.com/tldraw/status/1724892287304646868?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- small todo app https://twitter.com/Altimor/status/1725678615751438396
- [Nvidia ChipNeMo](https://twitter.com/DrJimFan/status/1724832446393598283), custom LLM trained on Nvidia’s internal data to generate and optimize software and assist human designers for GPU ASIC and Architecture engineers
- Lindy.ai https://twitter.com/Altimor/status/1721250514946732190
- Pplx api 
	- x
- DallE Party - gpt4v and dalle in a loop https://dalle.party/?party=0uKfJjQn

## misc and other discussions

- standard pretraining stack
	- Use flash attention 2, parallel attention and feedforward layers, rotary embeddings, pre-layer norm, and probably 8/3 h multipliers but that doesn't matter too much. Basically Mistral + parallel layers (they left a free +10% performance on the table). https://x.com/BlancheMinerva/status/1721380386515669209?s=20
	- Train on a large and diverse dataset, something like C4 along plus high quality known components (academic papers, books, code). Ideally you'd scrape these things freshly instead of using Pile / RP. You want to balance clean w/ diverse. Dedupe, but train for 4 epochs (3T+ total)
	- RLHF HISTORY Entangled Preferences: The History and Risks of Reinforcement Learning and Human Feedback
		- https://x.com/natolambert/status/1721250884481634420?s=20
- long context methods survey paper
	- https://arxiv.org/pdf/2311.12351.pdf
	- 
- MFU calculation
	- [Stas Bekman](https://twitter.com/StasBekman/status/1721207940168987113) - This study from 2020 publishes the actual achievable TFLOPS in half-precision for the high-end gpus. e.g., 88% for V100, 93% for A100. Showing that A100@BF16's peak performance is 290 and not 312 TFLOPS. So that means that when we calculate MFU (Model FLOPS Utilization) our reference point shouldn't be the advertised theoretical TFLOPS, but rather the adjusted achievable TFLOPS.
- other good reads
	- [Don't Build AI Products the Way Everyone Else Is Doing It](https://www.builder.io/blog/build-ai) ([builder.io](https://news.ycombinator.com/from?site=builder.io))[103 comments](https://news.ycombinator.com/item?id=38221552)|
- Galactica postmortem https://twitter.com/rosstaylor90/status/1724547381092573352
- [Notable resignation](https://twitter.com/ednewtonrex/status/1724902327151452486) from Stability AI Audio team, over copyright fair use training
- [beautiful illustrated piece](https://www.newyorker.com/humor/sketchbook/is-my-toddler-a-stochastic-parrot) on Stochastic Parrots by the New Yorker
- Google Switch C 1.6T MoE model (old, but being circulated)
	- https://news.ycombinator.com/item?id=38352794
- [$10m **Artificial Intelligence Mathematical Olympiad Prize** (AI-MO Prize)](https://aimoprize.com/) for AI models that can reason mathematically, leading to the creation of a publicly-shared AI model capable of winning a gold medal in the International Mathematical Olympiad (IMO).


## best memes

- https://twitter.com/EMostaque/status/1726591847282421871
- good meme material https://x.com/AISafetyMemes/status/1726595374402662497?s=20
- https://x.com/paularambles/status/1726606909069988141?s=20
- https://x.com/202accepted/status/1726505578405695713?s=20
- check latent space discord meme channel theres just too many
- 