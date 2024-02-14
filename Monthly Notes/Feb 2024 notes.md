
## openai

- memory and new controls https://news.ycombinator.com/item?id=39360724
- gpt-3.5-turbo-0125
	- The updated GPT-3.5 Turbo model is now available. It comes with 50% reduced input pricing, 25% reduced output pricing, along with various improvements including higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Returns a maximum of 4,096 output tokens.
- [chatgpt in apple vision pro](https://x.com/ChatGPTapp/status/1753480051889508485?s=20)
- lazy openai
	- https://www.reddit.com/r/OpenAI/comments/1aj6lrz/damned_lazy_ai/
- public appearances
	- logan on a big pod today https://www.lennyspodcast.com/inside-openai-logan-kilpatrick-head-of-developer-relations/

Frontier models
- RIP Bard [https://twitter.com/AndrewCurran_/status/1754546359460590002](https://twitter.com/AndrewCurran_/status/1754546359460590002 "https://twitter.com/AndrewCurran_/status/1754546359460590002")
	- https://blog.google/products/gemini/bard-gemini-advanced-app/
- Gemini ultra 1.0
	- [unclear advantage over gemini pro](https://www.youtube.com/watch?v=hLbIUQWxs6Y)
- https://twitter.com/evowizz/status/1753795479543132248

## models

- [Stable Cascade](https://news.ycombinator.com/item?id=39360106): a new arch vs stable diffusion
	-  Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a 1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the highly compressed latent space. 
	- Previous versions of this architecture, achieved a 16x cost reduction over Stable Diffusion 1.5.
	- Stable Cascade consists of three models: Stage A, Stage B and Stage C, representing a cascade for generating images, hence the name "Stable Cascade". Stage A & B are used to compress images, similarly to what the job of the VAE is in Stable Diffusion. However, as mentioned before, with this setup a much higher compression of images can be achieved. Furthermore, Stage C is responsible for generating the small 24 x 24 latents given a text prompt. The following picture shows this visually. Note that Stage A is a VAE and both Stage B & C are diffusion models.
	- For this release, we are providing two checkpoints for Stage C, two for Stage B and one for Stage A. Stage C comes with a 1 billion and 3.6 billion parameter version, but we highly recommend using the 3.6 billion version, as most work was put into its finetuning. The two versions for Stage B amount to 700 million and 1.5 billion parameters. Both achieve great results, however the 1.5 billion excels at reconstructing small and fine details. Therefore, you will achieve the best results if you use the larger variant of each. Lastly, Stage A contains 20 million parameters and is fixed due to its small size.
- [Nomic Embed](https://twitter.com/nomic_ai/status/1753082063048040829): Open source, open weights, open data
	- https://blog.nomic.ai/posts/nomic-embed-text-v1
	- Beats OpenAI text-embeding-3-small and Ada on short and long context benchmarks
- AI OLMo - 100% open-everything model
	- https://blog.allenai.org/olmo-open-language-model-87ccfc95f580
	- [released on magnet link](https://twitter.com/natolambert/status/1753063313351835941)
- [Natural-SQL-7B, a strong text-to-SQL model](https://github.com/cfahlgren1/natural-sql)
- [DeepSeekMath 7B](https://arxiv.org/abs/2402.03300) which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH
- [Presenting MetaVoice-1B](https://twitter.com/metavoiceio/status/1754983953193218193), a 1.2B parameter base model for TTS (text-to-speech). Trained on 100K hours of data. * Emotional speech in English * Voice cloning with fine-tuning * Zero-shot cloning for American & British voices * Support for long-form synthesis. Best part: Apache 2.0 licensed. 🔥
	- https://ttsdemo.themetavoice.xyz/
- [Reka Flash](https://twitter.com/YiTayML/status/1757115386829619534), a new state-of-the-art 21B multimodal model that rivals Gemini Pro and GPT 3.5 on key language & vision benchmarks 
- [SPIRIT-LM: Interleaved Spoken and Written Language Model](https://speechbot.github.io/spiritlm/index.html) from Meta
	- compare with: [LAION BUD-E](https://laion.ai/blog/bud-e/) - ENHANCING AI VOICE ASSISTANTS’ CONVERSATIONAL QUALITY, NATURALNESS AND EMPATHY
		- Right now (January 2024) we reach latencies of 300 to 500 ms (with a Phi 2 model). We are confident that response times below 300 ms are possible even with larger models like LLama 2 30B in the near future.


## open source tooling

- [Ollama openai compatibility APIs](https://news.ycombinator.com/item?id=39307330)
- [LlamaIndex v0.10](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8?source=collection_home---6------0-----------------------)

## product launches


## Misc reads

- learning
	- **[TPU-Alignment](https://github.com/Locutusque/TPU-Alignment)** - Fully fine-tune large models like Mistral-7B, Llama-2-13B, or Qwen-14B completely for free. on the weekly 20hrs of TPUv3-8 pod from Kaggle 
	- [undo llama2 safety tuning with $200 LoRA](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?)


## memes

- https://twitter.com/JackPosobiec/status/1753416551066181672 dignifAI
- https://www.goody2.ai/chat
- image of no elephant https://www.reddit.com/r/OpenAI/comments/1anm3p3/damn_sneaky/
	- source of meme: https://twitter.com/GaryMarcus/status/1755476468157833593
- apple vision pro https://discord.com/channels/822583790773862470/839660725252784149/1204047864096096328