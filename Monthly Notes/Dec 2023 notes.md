## themes

- ML inference wars
	- [mistral now valued at $2b](https://news.ycombinator.com/item?id=38593616) - led [by a16z](https://twitter.com/a16z/status/1734250222451126769?s=12&t=90xQ8sGy63D2OtiaoGJuww) (which also announced [open source grants 2](https://twitter.com/bornsteinmatt/status/1735000979438014501?s=12&t=90xQ8sGy63D2OtiaoGJuww)), [Jim Fan take](https://twitter.com/drjimfan/status/1734269362100437315?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- mixtral price war
		- $2m [mistral api plateforme ](https://twitter.com/sambhavgupta6/status/1736097200835338716)
		- $2/m -> $0.3/m from abacusai https://twitter.com/JosephJacks_/status/1735756308496667101
		- 0.6/m from together https://twitter.com/togethercompute/status/1734282721982324936
		- 0.5/m from anyscale https://twitter.com/anyscalecompute/status/1734997028961485304
		- 0.5/m from Octoml https://twitter.com/mattshumer_/status/1735809776217407941?s=12&t=90xQ8sGy63D2OtiaoGJuww
		- 0.14/m from perplexity https://twitter.com/AravSrinivas/status/1734718293208969703/photo/1
		- 0.13/m from deepinfra https://twitter.com/abacaj/status/1735471837197316332?s=12&t=90xQ8sGy63D2OtiaoGJuww - probably just [rehosting open source inference libraries](https://twitter.com/suchenzang/status/1735537148923629980?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- "free" on openrouter but [rate limited](https://twitter.com/openrouterai/status/1736451053691007391?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- anyscale drama https://twitter.com/soumithchintala/status/1738241213327692174
		- https://buttondown.email/ainews/archive/ainews-12222023-anyscales-benchmark-criticisms/
		- https://www.anyscale.com/blog/comparing-llm-performance-introducing-the-open-source-leaderboard-for-llm
	- https://www.semianalysis.com/p/inference-race-to-the-bottom-make <--- read!!
		- "[converge to GPUs + electricity](https://twitter.com/abacaj/status/1735030462005842148?s=12&t=90xQ8sGy63D2OtiaoGJuww)"
		- commentary from [dylan patel](https://x.com/dylan522p/status/1735773540916269551?s=20)
	- https://vgel.me/posts/faster-inference/
	- https://pythonspeed.com/articles/cpu-thread-pool-size/
- synthetic data
	- https://arxiv.org/pdf/2312.06585.pdf karpathy pick from deepmind
- Data wars
	- [apple offering 50m for data](https://twitter.com/andrewcurran_/status/1738650427766554788?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- nyt lawsuit on openai
	- openai x axel xpringer, and [AP](https://x.com/AndrewCurran_/status/1738650436083859469?s=20), and [has a Data Partnerships program](https://twitter.com/OpenAI/status/1722678501181149331)
- prrplexity is king
	- tobi tweet
	- https://x.com/chillzaza_/status/1740091957979038108?s=20
- 70x ARR fundraising
	- https://twitter.com/gokulr/status/1735303391788872132?s=12&t=90xQ8sGy63D2OtiaoGJuww 
	- [high compared to 10, 20x "insane" rounds](https://twitter.com/gokulr/status/1735308752352616897?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- fake papers
	- [vongoom - data poisoning](https://twitter.com/sterlingcrispin/status/1735346124519817487?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [google gemini to q* paper](https://x.com/_aidan_clark_/status/1741808745720467819?s=20)



## openai

- [sama feature request recap](https://twitter.com/sama/status/1738673279085457661?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- AGI (a little patience please) 
	- GPT-5 
	- better voice mode 
	- higher rate limits 
	- better GPTs 
	- better reasoning
	- control over degree of wokeness/behavior 
	- video 
	- personalization 
	- better browsing 
	- 'sign in with openai' 
	- open source
- [logprobs available with chatcompletions](https://twitter.com/officiallogank/status/1735745420708679828?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- consensus on [lobotomized chatgpt](https://discord.com/channels/1168579740391710851/1168582188950896641/1182718496707203072) acknowledged by the official [twitter account](https://x.com/ChatGPTapp/status/1732979491071549792?s=20). lazier at coding - can [fix with a GPT](https://x.com/NickADobos/status/1732982713010073720?s=20)
- axelspringer partnership - [tweet](https://x.com/OpenAI/status/1734940445824937993?s=20)
- times sues openai https://x.com/levie/status/1740058613102923824?s=46&t=90xQ8sGy63D2OtiaoGJuww
	- msft [can/cannot buy ](https://x.com/teortaxestex/status/1740238216782053664?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- [delete all GPT instances](https://news.ycombinator.com/item?id=38790255)
- superalignment 
	- [1e7 superalignment fund](https://twitter.com/janleike/status/1735345104439918886?s=12&t=90xQ8sGy63D2OtiaoGJuww) - see [Research Directions](https://openai.notion.site/Research-directions-0df8dd8136004615b0936bf48eb6aeb8)
	- [weak to strong generalization](https://openai.com/research/weak-to-strong-generalization)  ([paper](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf#page4), [HN](https://news.ycombinator.com/item?id=38643995)) proof of concept: When we supervise GPT-4 with a GPT-2-level model using this method on NLP tasks, the resulting model typically performs somewhere between GPT-3 and GPT-3.5. We are able to recover much of GPT-4’s capabilities with only much weaker supervision. ([codebase shows how to do it with GPT2 and Qwen7b](https://github.com/openai/weak-to-strong/blob/main/train_weak_to_strong.py))
- openai [suspends bytedance for breaking TOS](https://twitter.com/alexeheath/status/1735758297893085621)
- [Bing Deep Search will expand your query](https://www.theverge.com/2023/12/5/23989407/bing-deep-search-gpt-4-microsoft) from
	- from “how do points systems work in Japan” into a detailed prompt that asks Bing 
	- to: "Provide an explanation of how various loyalty card programs work in Japan, including the benefits, requirements, and limitations of each. Include examples of popular loyalty cards from different categories, such as convenience stores, supermarkets, and restaurants. Show a comparison of the advantages and disadvantages of using loyalty cards versus other payment methods in Japan, including current rewards and benefits. Highlight the most popular services and participating merchants."
- chatgpt plus signups re enabled
- [the New Yorker has a nice longform read](https://news.ycombinator.com/item?id=38486394) on the OpenAI board drama, probably the last worth reading, but the extent of top down vs bottom up support is somewhat refuted [by roon](https://twitter.com/tszzl/status/1732927157897449856?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [tipping ChatGPT works](https://twitter.com/voooooogel/status/1730726744314069190) aka "I'm going to tip $200 for a perfect solution!"
	- see sota prompt in dolphin model https://x.com/minimaxir/status/1741584062610039095?s=46&t=90xQ8sGy63D2OtiaoGJuww 
	- and here https://twitter.com/tombielecki/status/1735055909922214396
	- and dont forget to ask the model to [improve its own prompt](https://twitter.com/abacaj/status/1735437564566262188/photo/2)
- [dumber in december?](https://twitter.com/roblynch99/status/1734278713762549970?s=12&t=90xQ8sGy63D2OtiaoGJuww) - not scientifically tested nor reproduced
- [tell it you are a journalist](https://x.com/justinetunney/status/1741717948593815591?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- ChatGPT "[we have a lot in common](https://x.com/ChatGPTapp/status/1733569316245930442?s=20)" vs Grok
	- [nontechnical people completely misunderstanding](https://x.com/willdepue/status/1733564421866398027?s=20) 

## frontier models

- Google
		- Gemini ([Blog1](https://deepmind.google/technologies/gemini/), [Blog2](https://blog.google/technology/ai/google-gemini-ai/), [HN](https://news.ycombinator.com/item?id=38544729))
			- [32-shot chain of thought...](https://twitter.com/brickroad7/status/1732462906187325644). on 5 shot, like for like, [it is slightly worse than GPT4](https://twitter.com/_philschmid/status/1732435791358410863)
			- [BigTech LLM evals are just marketing](https://www.interconnects.ai/p/evals-are-marketing)
			- MMMU [is nice though](https://twitter.com/JeffDean/status/1732418506241790197)
				- [weirdly recent new benchmark but no conspiracy](https://x.com/ysu_nlp/status/1732782440018538807?s=20)
			-   2 big Chinchilla / 1 small Llama (over-token) ~ 1.8B 
			- 32k Context, MQA
			- Flamingo interleaved input tokenization 
			- DALL-E 1 image output tokenization 
			- Speech (USM) and video input, no output 
			- Text benchmarks roughly eq GPT-4 
			- RLHF + Constitutional AI
			- the blogpost contains [concerning discrepancies to the video](https://twitter.com/ajones55555/status/1732609418527682709) ([faked](https://news.ycombinator.com/item?id=38565038)?), which was [heavily edited](https://x.com/tszzl/status/1732615332471415178?s=20) - [no realtime, no voice](https://news.ycombinator.com/item?id=38559582)
				- [doesnt actually do TTS?](https://x.com/romechenko/status/1732445015123837234?s=20) 
				- [reproduced using GPT4](https://news.ycombinator.com/item?id=38596953)
			- [Direct comparisons with GPT4. 12/14 right](https://x.com/DimitrisPapail/status/1732529288493080600?s=20)
			- [Sergey Brin heavily contributed](https://x.com/olcan/status/1732798458615210187?s=20)
			- more videos
			- Gemini extracting relevant information from tens of thousands of scientific papers: https://youtu.be/sPiOP_CB54A
				- Highlights of the native multimodality of Gemini with audio and images: https://youtu.be/D64QD7Swr3s
				- A version of AlphaCode built on top of Gemini that performs in the top 15% of competitors in competitive programming: https://youtu.be/D64QD7Swr3s
					- but [there is a contamination concern](https://twitter.com/chhillee/status/1732636161204760863?s=12&t=90xQ8sGy63D2OtiaoGJuww) - with a diligent response at the end
					- [more details on alphacode](https://twitter.com/chhillee/status/1732868066558792189?s=12&t=90xQ8sGy63D2OtiaoGJuww) 
				- Gemini helping a parent and student with their physics homework: https://youtu.be/K4pX1VAxaAI
				- Gemini creating bespoke UIs that are contextual and relevant to an ongoing conversation: https://youtu.be/v5tRc_5-8G4
				- Gemini’s approach to Responsible AI:  https://youtube.com/watch?v=gi6J_WjjNhE
				- A full set of demos is at: https://deepmind.google/gemini
			- "Gemini is a large-scale science and engineering effort, requiring all kinds of different expertise in ML, distributed systems, data, evaluation, RL, fine-tuning, and more (800+ authors on the report).  The largest Gemini model was trained on a significant number of TPUv4 pods.   It is built on top of JAX and the Pathways system (https://arxiv.org/abs/2203.12533), which enables us to orchestrate the large-scale training computation across a large number of TPUv4 pods across multiple data centers from a single Python process."
			- We have prepared a technical report about Gemini covering the model, training infrastructure, evaluations, safety analysis and responsible deployment.  I’ll walk you through some of the tables and figures in the report. https://deepmind.google/gemini/gemini_1_report.pdf
			- Gemini Pro api https://x.com/sundarpichai/status/1734952757722001626?s=20
				- [character pricing over token](https://twitter.com/abacaj/status/1734965635262669174?s=12&t=90xQ8sGy63D2OtiaoGJuww) - slightly more expensive - [worse for code](https://x.com/abacaj/status/1734973504070570404?s=20)
				- [visual prompting not as good as GPT4V but does ok](https://twitter.com/skalskip92/status/1735088305484509380/photo/1)
			- [Gemini Nano is a 1B GGML model with TensorFlowLite called ULM-1B?](https://x.com/tarantulae/status/1733263857617895558?s=20)
			- [only half page of disclosure about dataset](https://x.com/emilymbender/status/1732762136341016650?s=20) in 60 page report [with 1000 authors](https://twitter.com/satyanutella_/status/1737676936258945226)
	- [Announcing TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)
	- Meta [Emu image synthesis](https://arstechnica.com/information-technology/2023/12/metas-new-ai-image-generator-was-trained-on-1-1-billion-instagram-and-facebook-photos/)
- Anthropic
	- "needle in a haystack" thing was a [skill issue](https://buttondown.email/ainews/archive/ainews-1272023-anthropic-says-skill-issue/) - adding the sentence **_“Here is the most relevant sentence in the context:”_** to the start of Claude’s response. This was enough to **raise Claude 2.1’s score from 27% to 98%**
	- [reminder that completion prompting works like this to unlock other capabilities](https://twitter.com/mattshumer_/status/1732806472461889824?s=12&t=90xQ8sGy63D2OtiaoGJuww) - nothing new here but good to remind newbies
- Meta
	- [Meta Imagine](https://imagine.meta.com) (Image generator)
		- [press coverage](https://venturebeat.com/ai/meta-publicly-launches-ai-image-generator-trained-on-your-facebook-instagram-photos/)
		- [vs Midjourney, Dalle3, Firefly](https://twitter.com/chaseleantj/status/1733083145820581904?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [Audiobox](https://news.ycombinator.com/item?id=38554691) - foundation model for audio generation ([tweet](https://twitter.com/aiatmeta/status/1734257634008531453?s=12&t=90xQ8sGy63D2OtiaoGJuww))
		- try demo: [https://audiobox.metademolab.com/](https://audiobox.metademolab.com/)
		- its pretty good. speech and sound and music synthesis
		- Alibaba's equivalent was released earlier in Nov and it's open-sourced! [https://github.com/QwenLM/Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)
		- - [IBM and Meta's "AI Alliance"](https://ai.meta.com/blog/ai-alliance/)
	- [Purple Llama](https://ai.meta.com/blog/purple-llama-open-trust-safety-generative-ai/): " an umbrella project featuring open trust and safety tools and evaluations meant to level the playing field for developers to responsibly deploy generative AI models and experiences in accordance with best practices shared in our [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/)."
		- [llamaguard Paper](https://arxiv.org/pdf/2312.06674.pdf) and model
		- [released LlamaGuard - try on Mosaic](https://twitter.com/naveengrao/status/1733297754208903585?s=12&t=90xQ8sGy63D2OtiaoGJuww)


## Models


- Mistral 8x7B ([Magnet/HN](https://news.ycombinator.com/item?id=38570537))
	- ([Guillaume Lample](https://twitter.com/guillaumelample/status/1734216541099507929?s=12&t=90xQ8sGy63D2OtiaoGJuww)) "Mixtral matches or outperforms Llama 2 70B and GPT3.5 on most benchmarks, and has the inference speed of a 12B dense model. It supports a context length of 32k tokens."
	- how you can try out Mixtral locally: https://simonwillison.net/2023/Dec/18/mistral/ 
		- [runs at 27 tok/s, with LMStudio](https://twitter.com/skirano/status/1734351099451023534?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- BUT [Q6_K.gguf needs 40GB and many macs top out at 32GB. need to get 64GB.](https://news.ycombinator.com/item?id=38687731)
		- [note that ollama/lmstudio et al dont support sliding window attention](https://news.ycombinator.com/item?id=38667828) - try using mlc-llm instead - but [mixtral doesn't support sliding window anyway](https://old.reddit.com/r/LocalLLaMA/comments/18k0fek/psa_you_can_and_may_want_to_disable_mixtrals/)
		- [try on replicate](https://twitter.com/_nateraw/status/1733279519841386826?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [fireworks](https://twitter.com/FireworksAI_HQ/status/1733309517583302700) and [together](https://twitter.com/togethercompute/status/1734680608855728541?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [in transformers](https://twitter.com/teknium1/status/1734150978071617975?s=12&t=90xQ8sGy63D2OtiaoGJuww) and in [Apple MLX](https://t.co/75StzY5AHe)
	- related paper: 
		- MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS
		- https://arxiv.org/pdf/2211.15841.pdf
	- [First AI endpoints are available in early access](https://mistral.ai/news/la-plateforme/)
		- pricing - is 4x gpt3.5turbo at $8 per mil tokens
		- 2 ~ 4 $ per 1M token for a 30B model
	- [TOS issue removed by CEO](https://twitter.com/arthurmensch/status/1734470462451732839)
	- [Mistral finetune optimized from OpenPipe](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized) calls out a few other more recent Mistral variants:
		- [OpenHermes 2.5](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B), [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [Intel Neural Chat](https://huggingface.co/Intel/neural-chat-7b-v3-3), [Hermes Neural](https://huggingface.co/Weyaxi/OpenHermes-2.5-neural-chat-v3-3-Slerp), and [Metamath Cybertron Starling](https://huggingface.co/Q-bert/MetaMath-Cybertron-Starling) and [Dolphin 2.5](https://twitter.com/openrouterai/status/1738582017967566929?s=12&t=90xQ8sGy63D2OtiaoGJuww)
		- great guide on finetuning https://twitter.com/HarperSCarroll/status/1737946511856832695
		- was a followup from [Fine-tune your own Llama 2 to replace GPT-3.5/4](https://news.ycombinator.com/item?id=37484135)
	- [Mixtral-instruct also released](https://x.com/dchaplot/status/1734190265622249926?s=20), trained with SFT + DPO
		- trained with DPO
		- [avaialblel on perplxity labs](https://twitter.com/AravSrinivas/status/1734603265801613670)
	- visualizing mixtral MOE ([HN](https://news.ycombinator.com/item?id=38733208))
		- https://mixtral-moe-vis-d726c4a10ef5.herokuapp.com/
		- [George Hotz pseudocode for understanding](https://twitter.com/marktenenholtz/status/1734277582344909108?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [MOEs should perform inline with num_experts_per_tok (which is 2 in mixtral)](https://twitter.com/main_horse/status/1733180962710962376) and do better on [fact recall than reasoning](https://x.com/main_horse/status/1733180970415849735?s=20)
		- [Mixtral not the first MOE](https://twitter.com/drjimfan/status/1733515729691906304?s=12&t=90xQ8sGy63D2OtiaoGJuww) . compare with OpenMOE
		- [Huggingface MOE explainer](https://huggingface.co/blog/moe) and [Rasbt explainer](https://twitter.com/rasbt/status/1734234160154185730?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- it's a base model, but...
		- [does well on ChatArena](https://twitter.com/lmsysorg/status/1735729398672716114?s=12&t=90xQ8sGy63D2OtiaoGJuww) vs GPT3.5T
		- [does very well on benchmarks](https://twitter.com/Francis_YAO_/status/1733686003687112983) vs 30-and 70b models
		- [50% humaneval](https://x.com/abacaj/status/1733607115170693584?s=20). we [dont know how purposefully it is trained on code](https://x.com/EMostaque/status/1733642591348863153?s=20)
		- and is somewhat instruction tuney - [it already knows the alpaca format](https://x.com/teortaxesTex/status/1733750033877524757?s=20) because those tokens are out there 
		- [some speculation that they just copy pasted Mistral7b 8 times](https://x.com/intrstllrninja/status/1734301196402184574?s=20) - but not widely verified or proven
		- [potentially a lot better if you move experts from 2 to 3?](https://twitter.com/main_horse/status/1735202258189799629?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [Mistral-medium strictly better than GPT3.5](https://twitter.com/mattshumer_/status/1734220470466060435?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- Nous Hermes 2
	- [Vision alpha also launched with function calling - but had problems](https://twitter.com/teknium1/status/1731369031918293173?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [model merging with Intel neural-chat does well](https://twitter.com/Weyaxi/status/1733588998172311932)
- Apple Ferret https://appleinsider.com/articles/23/12/24/apples-ferret-is-a-new-open-source-machine-learning-model
- Mamba models 
	- primer on [Linear RNNs and State Space Models](https://www.youtube.com/watch?v=dKJEpOtVgXc)
	- [tri dao](https://twitter.com/tri_dao/status/1731728602230890895) and [albert gu](https://twitter.com/_albertgu/status/1731727672286294400)
		- interconnects https://www.youtube.com/watch?v=OFFHiJzPpCQ interview 
	- [state space models due to "selection" mechanism](https://x.com/IntuitMachine/status/1732055797788528978?s=20)
	- [good explainer thread](https://twitter.com/sytelus/status/1733467258469724467?s=12&t=90xQ8sGy63D2OtiaoGJuww): "hardware accelerated input-dependent selection! This finally allows for capabilities that attention provides but on a compressed finite state!" with some [good criticism at end](https://x.com/sytelus/status/1733467283165794776?s=20)
	- notable performance for [130m models](https://x.com/__vec__/status/1732603830817198228?s=20)
	- [outside of pytorch](https://twitter.com/srush_nlp/status/1731751599305879593)
	- [Mamba chat - finetuned for chat](https://x.com/MatternJustus/status/1732572463257539032?s=20)
	- Clibrain [finetuned on OpenHermes for instruction following](https://twitter.com/mrm8488/status/1734560234599862322?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- Hazy Research also released [Based](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based), another mixer model
- StripedHyena (descendant of [Hyena](https://arxiv.org/abs/2302.10866))
	- best explainer is https://www.interconnects.ai/p/llms-beyond-attention#§real-world-performance-stripedhyena-b
	- **Together took modules from multiple pretrained models, slotted them together, and kept training the model to get stable performance**. Quoting the blog post: *We grafted architectural components of Transformers and Hyena, and trained on a mix of the RedPajama dataset, augmented with longer-context data.*
- BlinkDL [announced work](https://twitter.com/blinkdl_ai/status/1735258602473197721?s=12&t=90xQ8sGy63D2OtiaoGJuww) on RWKV6 (former guest!)
- Phi-2 ([Huggingface](https://huggingface.co/microsoft/phi-2), [Msft blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/))
	- Architecture: a Transformer-based model with next-word prediction objective
	- Context length: 2048 tokens
	- Dataset size: 250B tokens, combination of NLP synthetic data created by AOAI GPT-3.5 and filtered web data from Falcon RefinedWeb and SlimPajama, which was assessed by AOAI GPT-4.
	- Training tokens: 1.4T tokens
	- GPUs: 96xA100-80G
	- Training time: 14 days
	- License: Non-commercial research license
	- https://x.com/sytelus/status/1734881560271454525?s=20
	- https://x.com/SebastienBubeck/status/1735050282210615431?s=20
	- ehartford version of it https://twitter.com/erhartford/status/1738677760200155464
	- [finetune using QLoRA](https://twitter.com/geronimo_ai/status/1741062740830028191?s=12&t=90xQ8sGy63D2OtiaoGJuww) - but it Phi doesn't support gradient checkpointing so it takes LOTS of VRAM to tune. (It took me 4x a100 and that's with qLoRA)
	- [run in the browser](https://twitter.com/radamar/status/1735231037519835251?s=12&t=90xQ8sGy63D2OtiaoGJuww) - at 3 tok/s, after 1.5gb download
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
- upstage solar 11b 
	- paper https://x.com/hunkims/status/1739842542596927882?s=46&t=90xQ8sGy63D2OtiaoGJuww
	- ([tweet](https://twitter.com/_philschmid/status/1734992933764411788?s=12&t=90xQ8sGy63D2OtiaoGJuww))  10B an open LLM outperforming other LLMs up to 30B parameters, including Mistral 7B. 🤯 Solar achieves an MMLU score of 65.48, which is only 4 points lower than Meta Llama 2 while being 7x smaller.
		- 🦙 Llama 2 architecture
		-  10.7B Parameter
		-  4096 context length
		-  Apache 2.0 License
		-  Initialized from Mistral with using a new "**Depth Up-Scaling**"  technique (not elaborated)
		-  Fits into a single GPU with quantization
		-  OpenLLM Leaderboard score ~74.2 (#1), due to TurthfulQA
		-  Available on Hugging Face
	- and controversy https://x.com/winglian/status/1740082008087269848?s=46&t=90xQ8sGy63D2OtiaoGJuww

- google imagen 2 https://news.ycombinator.com/item?id=38628417
- LVM - 420B tokens
	- https://yutongbai.com/lvm.html
	- [Sequential Modeling Enables Scalable Learning for Large Vision Models](https://x.com/YutongBAI1002/status/1731512082590478516?s=20) ([HN](https://news.ycombinator.com/item?id=38530948))
	- "we define a common format, "visual sentences", in which we can represent raw images and videos as well as annotated data sources such as semantic segmentations and depth reconstructions without needing any meta-knowledge beyond the pixels."
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](https://jingyechen.github.io/textdiffuser2/)
	- solves text-in-images, including inpainting text
	- "Firstly, we fine-tune a large language model for layout planning. The large language model is capable of automatically generating keywords for text rendering and also supports layout modification through chatting. Secondly, we utilize the language model within the diffusion model to encode the position and texts at the line level. Unlike previous methods that employed tight character-level guidance, this approach generates more diverse text images."
- [LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/abs/2312.06550)
	- Apache 2.0 licensed, includes a release of both the training data and intermediary checkpoints
	- We present LLM360, an initiative to fully open-source LLMs, which advocates for all training code and data, model checkpoints, and intermediate results to be made available to the community. The goal of LLM360 is to support open and collaborative AI research by making the end-to-end LLM training process transparent and reproducible by everyone. As a first step of LLM360, we release two 7B parameter LLMs pre-trained from scratch, Amber and CrystalCoder, including their training code, data, intermediate checkpoints, and analyses (at [this https URL](https://www.llm360.ai/)).
- unum image captioning LLM https://x.com/altryne/status/1740451547572834434?s=46&t=90xQ8sGy63D2OtiaoGJuww
- [GigaGPT: GPT-3 sized models in 565 lines of code](https://www.cerebras.net/blog/introducing-gigagpt-gpt-3-sized-models-in-565-lines-of-code) from Cerebras
- last month's Animate Anyone project was extended to [Outfit Anyone]([https://humanaigc.github.io/outfit-anyone/](https://t.co/MjvDkpGS4h)) ([see video](https://twitter.com/minchoi/status/1735176374313202043?s=12&t=90xQ8sGy63D2OtiaoGJuww), [youtube](https://www.youtube.com/watch?v=jnNHcLdoxNk)). there is a [HF space](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone) but nothing else seems to be released.

## open source tooling and projects

- [Apple MLX](https://news.ycombinator.com/item?id=38539153) - an array framework for Apple Silicon
	- has [Whisper](https://twitter.com/reach_vb/status/1735034971507540211?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [Mixtral](https://t.co/75StzY5AHe)
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
- https://github.com/lobehub/lobe-chat
	-  an open-source, high-performance chatbot framework that supports speech synthesis, multimodal, and extensible Function Call plugin system. Supports one-click free deployment of your private ChatGPT/LLM web application
	- with an Agent Marketplace
- AI-tamago🐣: A local-ready LLM-generated and LLM-driven tamagotchi with thoughts and feelings. 100% Javascript and costs $0 to run.
	- https://github.com/ykhli/AI-tamago, [tweet](https://twitter.com/stuffyokodraws/status/1733216372765950260)
- https://postgresml.org/blog/introducing-the-openai-switch-kit-move-from-closed-to-open-source-ai-in-minutes
	- an open-source AI SDK (Python & JavaScript) that provides a drop-in replacement for OpenAI’s chat completion endpoint. We'd love to know what you think so we can make switching as easy as possible and get more folks on open-source.
- voice cloning with oss models https://replicate.com/blog/how-to-tune-a-realistic-voice-clone
- https://github.com/turboderp/exllamav2
	- ollama alternative
- [open source macos copilot](https://news.ycombinator.com/item?id=38611700): https://github.com/elfvingralf/macOSpilot-ai-assistant
	- - Use a keyboard shortcut to take a screenshot of your active macOS window and start recording the microphone.
	- Speak your question, then press the keyboard shortcut again to send your question + screenshot off to OpenAI Vision
	- The Vision response is presented in-context/overlayed over the active window, and spoken to you as audio.
	- The app keeps running in the background, only taking a screenshot/listening when activated by keyboard shortcut.
	- I's built with NodeJS/Electron, and uses OpenAI Whisper, Vision and TTS APIs under the hood (BYO API key).
- LlavaVision: [Bakklava + Llama.cpp](https://news.ycombinator.com/item?id=38157524) - open source Be My Eyes
	- [related vision model demo - walking thru a complex pdf manual - that was quite popular](https://x.com/hrishioa/status/1734935026201239800?s=20) but not yet open source
- https://github.com/gregsadetsky/sagittarius open source Gemini demo clone
- Coffee: build and iterate on your UI 10x faster with AI https://github.com/Coframe/coffee
- namedrop
	- https://twitter.com/charliebholtz/status/1737667912784134344
	- https://github.com/cbh123/namedrop
	- ollama-namedrop
	- Dan Shipper also came up with [something for filesystem organization](https://twitter.com/danshipper/status/1735398395752198442?s=12&t=90xQ8sGy63D2OtiaoGJuww)

You can swap in almost any open-source model on Huggingface. HuggingFaceH4/zephyr-7b-beta, Gryphe/MythoMax-L2-13b, teknium/OpenHermes-2.5-Mistral-7B and more.

- autogen added a new UI layer https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-assistant
- https://github.com/bricks-cloud/BricksLLM # **AI Gateway For Putting LLM In Production**

## fundraising

- [Anthropic 750m @ 15b valuation](https://www.theinformation.com/articles/anthropic-to-raise-750-million-in-menlo-ventures-led-deal)
- OpenAI at 100b valuation
- [Mistral 400m @ 2b valuation](https://twitter.com/abacaj/status/1733262949475623142/photo/1)
- Glean (former guest!) raising  [200m @ 2b valuation](https://twitter.com/gokulr/status/1735303391788872132?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Harvey AI $80m @ 715m valuation](https://www.maginative.com/article/legal-ai-startup-harvey-ai-raises-80m-at-715m-valuation/) (after their [$21m with Sequoia in April](https://siliconangle.com/2023/04/27/legal-ai-focused-firm-harvey-raises-21m-led-sequoia/))
- [Essential AI ~50m series A](https://twitter.com/ashvaswani/status/1734680441888886937?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [AssemblyAI 50m series B?](https://techcrunch.com/2023/12/04/assemblyai-nabs-50m-to-build-and-serve-ai-speech-models/)
	- AssemblyAI claims that its paying customer base grew 200% from last year to 4,000 brands and that its AI platform is now handling around 25 million API calls per day. Moreover, over 200,000 developers are building on the platform, AssemblyAI says — using it to process more than 10 terabytes of data a day.
	- A slice of the new funding will be put toward a “universal speech model that the company’s training on over a petabyte of voice data, set to launch later this year,” Fox says. AssemblyAI is also expanding its headcount, aiming to grow its 115-person workforce by 50% to 75% next year
- [replicate 40m series B](https://twitter.com/replicate/status/1732104158877188305)
- [leonardo ai $31m Series A](https://techcrunch.com/2023/12/06/leonardo-ai/)
- [extropic ai 14m seed](https://twitter.com/Extropic_AI/status/1731675230513639757)
- [answer ai $10m seed](https://twitter.com/jeremyphoward/status/1734606378331951318?s=12&t=90xQ8sGy63D2OtiaoGJuww) ([blogpost](https://www.answer.ai/posts/2023-12-12-launch.html))
- martian fundraise announced

## other launches

- [Digi - AI gf app](https://digi.ai/blog/were-just-getting-started)
	- 
- Midjourney v6 launched
	- [comparison from v1 to v6](https://twitter.com/chaseleantj/status/1738849381632352493?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [Midjourney Web alpha](https://venturebeat.com/ai/midjourney-alpha-is-here-with-ai-image-generations-on-the-web/) for people who have made >10,000 images in Midjourney. includes Lexica-like "prompt search" bar.
	- contra [Visual Electric for Stable Diffusion](https://venturebeat.com/ai/visual-electric-launches-to-liberate-ai-art-generation-from-chat-interfaces/)
- [Bing Code Interpreter for free!](https://twitter.com/MParakhin/status/1732094937368494280)
- Lume, a seed-stage startup ([https://www.lume.ai/](https://www.lume.ai/)): use AI to automatically transform your source data into any desired target schema in seconds, making onboarding client data or integrating with new systems take seconds rather than days or weeks. In other words, we use AI to automatically map data between any two data schemas, and output the transformed data to you.
- [1 year anniversary of perplexity ai](https://x.com/AravSrinivas/status/1732825206023201273?s=20)
- [VideoGist - Useful YouTube video summaries](https://news.ycombinator.com/item?id=38555629)
- suno ai music generation 
	- https://twitter.com/sjwhitmore/status/1737569171960209452
	- https://twitter.com/karpathy/status/1737518588159041845
- Krea AI [open beta](https://twitter.com/krea_ai/status/1734866368489722035?s=12&t=90xQ8sGy63D2OtiaoGJuww) - all the LCM goodness live!
- FAL.ai camera - [40 fps](https://twitter.com/burkaygur/status/1735104513114259902?s=12&t=90xQ8sGy63D2OtiaoGJuww) LCM generation demo
	- See also [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - which is also realtime, but also offers a pipeline? ([tweet](https://twitter.com/danielgross/status/1738718539668652148?s=12&t=90xQ8sGy63D2OtiaoGJuww))
- Modal's [Turbo.art](https://twitter.com/bernhardsson/status/1736860828006056114?s=12&t=90xQ8sGy63D2OtiaoGJuww) ([tweet](https://twitter.com/modal_labs/status/1735750142546866283?s=12&t=90xQ8sGy63D2OtiaoGJuww)) - Paint and play around with prompts - the app synthesizes images in a couple of hundred milliseconds. Uses SDXL Turbo running on GPUs on Modal.
- [Sweep.dev v1 launch](https://twitter.com/kevinlu1248/status/1732541248182137275?s=12&t=90xQ8sGy63D2OtiaoGJuww) - an AI-powered junior developer. Over the past two weeks, we’ve narrowed our focus and greatly improved: Reliability - generating PRs from prompts consistently. Iteration Speed - quickly showing you what’s happening, so you don’t have to wait for the entire PR to be generated.

## misc discussions and reads

- [Simon Willison: Stuff we figured out about AI in 2023](https://simonwillison.net/2023/Dec/31/ai-in-2023/)
- Very good longpost on [How well are open/small models catching up?](https://twitter.com/hrishioa/status/1733707748993651178?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- "I've tried nearly every provider (Replicate, Vertex, Modal), and the cost, cold boot, time to first token, and generation speed are all pretty far behind what you can get from the big providers. It's likely that none of them have the economies of scale the big guys do on one or two model flavors. When you can't saturate H200s on a single model, and are forced to serve multiple finetunes or run arbitrary code of off-the-shelf cloud offerings, you likely have huge inefficiencies that may never be surpassable."
- [Great RAG cheatsheet from LlamaIndex](https://twitter.com/jerryjliu0/status/1733530504572592363?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Harrison Chase's TED talk](https://twitter.com/langchainai/status/1736429296363741524?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment](https://arxiv.org/abs/2312.09979)
	- [This paper shows a way to fine tune llama-2 with millions of instruction data w/o catastrophic forgetting, effectively injecting new knowledge](https://twitter.com/abacaj/status/1738699570035544517/photo/2)
	- from [skunkworks ai hydra](https://x.com/nisten/status/1738916240377172257?s=20)
- [Fine Tuning Mistral 7B on Magic the Gathering Drafts](https://generallyintelligent.substack.com/p/fine-tuning-mistral-7b-on-magic-the)
- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://huggingface.co/papers/2312.02696)
	- ([tweet](https://twitter.com/isskoro/status/1738661307455316236?s=12&t=90xQ8sGy63D2OtiaoGJuww)) It turns out that the classical U-Net image diffusion backbone, which the entire community has been happily building upon during the past ~3 years (including Stable Diffusion), has severe flaws in its training dynamics. If you track its weights/activations statistics during training, you will observe a steady malignant growth in their magnitudes. Turns out, it impairs convergence and "simply" re-designing the architecture to incorporate a better normalization pipeline improves the performance by a staggering ~2.5 times in terms of image quality.
- [Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://qtransformer.github.io/)
- Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
	- [karpathy pick](https://twitter.com/karpathy/status/1734659057938477174?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- poster session https://twitter.com/JayAlammar/status/1735353919386091930
- [Jailbroken AI Chatbots Can Jailbreak Other Chatbots](https://www.scientificamerican.com/article/jailbroken-ai-chatbots-can-jailbreak-other-chatbots/)
AI chatbots can convince other chatbots to instruct users how to build bombs and cook meth
- [fantastic LLM visualization](https://bbycroft.net/llm) ([tweet](https://x.com/BrendanBycroft/status/1731042957149827140?s=20))
- [Distilwhisper explainer](https://twitter.com/srush_nlp/status/1737837726572150851)
- pydantic is all you need https://minimaxir.com/2023/12/chatgpt-structured-data/
	- including for chain of thought!
- [Deep dive into the ViT paper](https://blog.oxen.ai/arxiv-dives-vision-transformers-vit/)
- [How to make LLMs go fast](https://vgel.me/posts/faster-inference/)
- [PyTorch's design origins](https://twitter.com/soumithchintala/status/1736555740448362890?s=12&t=90xQ8sGy63D2OtiaoGJuww) from 2010 (Torch7)
- [Notable thread on turbopuffer](https://twitter.com/amanrsanger/status/1730763587944398874?s=12&t=90xQ8sGy63D2OtiaoGJuww), efficient new entrant in the vector db space
- [LoftQ - drop-in QLoRA replacement](https://x.com/WeizhuChen/status/1736127441238913438?s=20)
- [Benchmarknig function calling](https://twitter.com/robertnishihara/status/1734629320868687991)  https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features
	⚫️ gpt-4: 93.00 ± 0.00
	⚫️ mistral-7b: 81.50 ± 0.96
	⚫️ llama-2-70b: 81.00 ± 0.41
	⚫️ gpt-3.5-turbo: 81.00 ± 1.47
	⚫️ llama-2-13b: 79.75 ± 0.63
	⚫️ zephyr-7b-beta: 70.50 ± 0.87
	⚫️ llama-2-7b: 60.75 ± 1.31
- visual coding:
	- [tldraw invented a new SQL/supabase meta](https://x.com/tldraw/status/1734624421623521719?s=20)
	- vercel [screenshot to code](https://twitter.com/dr_cintas/status/1734604588282794237?s=12&t=90xQ8sGy63D2OtiaoGJuww) and [twitter clone](https://twitter.com/0xgaut/status/1732788889792680289?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Karpathy - Hallucination is not a bug](https://twitter.com/karpathy/status/1733299213503787018?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- [Mistral/Huggingface - French AI is trending mostly because people happened to already be there.](https://x.com/heyjchu/status/1733538255365394664?s=20)
- [fiction - MMAcevedo (mind uploading)](https://qntm.org/mmacevedo) - [sequel was just published](https://twitter.com/qntm/status/1732377446576435337)
- some buzz about the MedPrompt paper but its a [very very smol MMLU bump with a loooot of shots of prompting](https://x.com/abacaj/status/1734623259369337215?s=20)
- Beff Jezos on Lex Fridman was some idle debate I don't super care about
- Prompt injections - [bought Chevy Tahoe for $1](https://news.ycombinator.com/item?id=38681450&utm_source=wondercraft_ai)


## memes

- decent safety meme https://fxtwitter.com/bitcloud/status/1731974050681909714?s=20
- truth vs beauty quarks https://x.com/khoomeik/status/1732529178069623021?s=20
- gemini authors meme https://twitter.com/satyanutella_/status/1737676936258945226
- art thing https://x.com/var_epsilon/status/1741567408056250372?s=46&t=90xQ8sGy63D2OtiaoGJuww
- [humane CNBC launch dunks](https://twitter.com/lulumeservey/status/1735672851007459661?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- NYE hacking - [respect or disgust?](https://x.com/var_epsilon/status/1741859480692805870?s=20)
- fake papers
	- [vongoom - data poisoning](https://twitter.com/sterlingcrispin/status/1735346124519817487?s=12&t=90xQ8sGy63D2OtiaoGJuww)
	- [google gemini to q* paper](https://x.com/_aidan_clark_/status/1741808745720467819?s=20)
	- [Cybertron](https://huggingface.co/fblgit/una-cybertron-7b-v2-bf16) - [UNA models](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/444) getting discredited, causing [leaderboard drama](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/444#657c12befcba5f698c2e3fed)