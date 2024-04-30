
## top themes

- - [became the #6 model on lmsys](https://twitter.com/lmsysorg/status/1777630133798772766?t=90xQ8sGy63D2OtiaoGJuww)and top open model, beating mistral large and qwen, but behind claude sonnet, gemini pro, and gpt4t


## openai

- harvey case study https://x.com/gabepereyra/status/1775207692841488542?s=20
	- - 93% preferred vs. ChatGPT by BigLaw attorneys - 87% more accurate case citations
- more improvements to OpenAI's Fine-Tuning API & additional info on its Custom Models program [https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program "https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program")
	- Epoch-based Checkpoint Creation: Automatically produce one full fine-tuned model checkpoint during each training epoch, which reduces the need for subsequent retraining, especially in the cases of overfitting
	- Comparative Playground: A new side-by-side Playground UI for comparing model quality and performance, allowing human evaluation of the outputs of multiple models or fine-tune snapshots against a single prompt
	- Third-party Integration: Support for integrations with third-party platforms (starting with Weights and Biases this week) to let developers share detailed fine-tuning data to the rest of their stack
	- Comprehensive Validation Metrics: The ability to compute metrics like loss and accuracy over the entire validation dataset instead of a sampled batch, providing better insight on model quality
	- Hyperparameter Configuration: The ability to configure available hyperparameters from the Dashboard (rather than only through the API or SDK) 
	- Fine-Tuning Dashboard Improvements: Including the ability to configure hyperparameters, view more detailed training metrics, and rerun jobs from previous configurations
- minor technical stuff
	- [batch endpoint](https://twitter.com/OpenAIDevs/status/1779922566091522492) - Just upload a file of bulk requests, receive results within 24 hours, and get 50% off API prices
	- `tool_choice: required` uses [constrained sampling](https://twitter.com/gdb/status/1784990428854391173) for openai funciton calling
	- [GPT4V GA](https://twitter.com/OpenAIDevs/status/1777769463258988634) now also uses JSON mode and function calling - useases devin, healthify snap, tldraw makereal. openai account calls it "[majorly improved](https://x.com/OpenAI/status/1777772582680301665)" GPT4T
		- specifically [reasoning has been further improved](https://x.com/polynoamial/status/1777809000345505801?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the). and [math](https://twitter.com/owencm/status/1777770827985150022) 
		- [cursor has better comparison](https://twitter.com/cursor_ai/status/1777886886884986944?t=6FDPaNxZcbSsELal6Sv7Ug)
		- GPT4T upgrade - [less verbose, better at codegen](https://twitter.com/gdb/status/1778126026532372486?t=6FDPaNxZcbSsELal6Sv7Ug)
- rumors
	- [NYT says openai scraped 1m hrs of youtube data](https://www.theverge.com/2024/4/6/24122915/openai-youtube-transcripts-gpt-4-training-data-google)

## frontier models

- Anthropic
	- [Claude - tool use now in beta](https://twitter.com/AnthropicAI/status/1775979802627084713)
- Google
	- gemini unerstand audio, uses unlimited files, offers json mode, no more waitlist https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww
		- https://x.com/OfficialLoganK/status/1777733743303696554
	- [codegemma](https://x.com/_philschmid/status/1777673558874829090) - 2b (27% humaneval) & 7b (52% humaneval) with 8k context - 500b extra tokens

## open models

- Llama 3
	- [top 5 in Lmsys, but also tied for first in English](https://x.com/lmsysorg/status/1782483701710061675?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- Cohere Command R+: [@cohere](https://twitter.com/cohere/status/1775878850699808928) released Command R+, a 104B parameter model with 128k context length, open weights for non-commercial use, and strong multilingual and RAG capabilities. It's available on the [Cohere playground](https://twitter.com/cohere/status/1775878883268509801) and [Hugging Face](https://twitter.com/osanseviero/status/1775882744792273209). [Aidan tweet](https://twitter.com/aidangomez/status/1775878606108979495)
	-   **Optimized for RAG workflows**: Command R+ is [optimized for RAG](https://twitter.com/aidangomez/status/1775878606108979495), with multi-hop capabilities to break down complex questions and strong tool use. It's integrated with [@LangChainAI](https://twitter.com/cohere/status/1775931339361149230) for building RAG applications.
	-   **Multilingual support**: Command R+ has [strong performance](https://twitter.com/seb_ruder/status/1775882934542533021) across 10 languages including English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Arabic, and Chinese. The SonnetTokenizer is [efficient for non-English text](https://twitter.com/JayAlammar/status/1775928159784915229).
	- [became the #6 model on lmsys ](https://twitter.com/lmsysorg/status/1777630133798772766?t=90xQ8sGy63D2OtiaoGJuww)and top open model, beating mistral large and qwen, but behind claude sonnet, gemini pro, and gpt4t
- Mistral 8x22B
- Phi-3 ([HN, Technical Report](https://news.ycombinator.com/item?id=40127806), [sebastian bubeck short video](https://twitter.com/SebastienBubeck/status/1782627991874678809?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1782627991874678809%7Ctwgr%5E507304ee4fbb7b0a8a9c60b9bb5711109bde1d41%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.emergentmind.com%2Fpapers%2F2404.14219))
	- phi-3-mini: 3.8B model trained on 3.3T tokens rivals Mixtral 8x7B and GPT-3.5
	- phi-3-medium: 14B model trained on 4.8T tokens w/ 78% on MMLU and 8.9 on MT-bench
	- phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench
	- The mobile-friendly design is quantized to a 4-bit model, reducing its memory footprint to approximately 1.8GB
	- The innovation lies entirely in our dataset for training, a scaled-up version of the one used for phi-2, composed of heavily filtered web data and synthetic data.
	- We also provide some initial parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and 78% on MMLU, and 8.7 and 8.9 on MT-bench)
- [Stable Audio 2.0](https://x.com/StabilityAI/status/1775501906321793266?s=20) - a new model capable of producing high-quality, full tracks with coherent musical structure up to three minutes long at 44.1 kHz stereo from a single prompt.
- Qwen 1.5-32B-Chat ([HF](https://huggingface.co/Qwen/Qwen1.5-32B-Chat)) - the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:
	- 8 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B and 72B dense models, and an MoE model of 14B with 2.7B activated;
	- Significant performance improvement in human preference for chat models;
	- Multilingual support of both base and chat models;
	- Stable support of 32K context length for models of all sizes
- IDEFICS 2 https://x.com/ClementDelangue/status/1779925711991492760

## open source tooling

- LMsys arena-hard: https://twitter.com/lmsysorg/status/1782179997622649330
	-  a pipeline to build our next generation benchmarks with live Arena data.
	- Significantly better separability than MT-bench (22.6% -> 87.4%)
	- Highest agreement to Chatbot Arena ranking (89.1%)
	- Fast & cheap to run ($25)
	- Frequent update with live data
	- We propose to use Confidence Intervals via Bootstrapping to calculate below two metrics:
	- Agreement with human: does benchmark have high agreement to human preference?
	- Separability: can benchmark confidently separate models?
	- Arena-hard achieves the highest on both, serving as a fast proxy to Chatbot Arena ranking.
	- How does Arena-hard pipeline work?
		1) Input: 200K Arena user prompts
		2) Topic modeling to ensure diversity 
		3) Key criteria (e.g., domain knowledge, problem-solving) to select high quality topic clusters
	1) Result: 500 challenging benchmark prompts.
- [LangChain - ToolCallingAgent](https://twitter.com/LangChainAI/status/1778465775034249625?utm_source=ainews&utm_medium=email)
	- A standard `bind_tools` method for attaching tools to a model that handles provider-specific formatting.
- https://github.com/GregorD1A1/TinderGPT
- https://github.com/princeton-nlp/SWE-agent
- https://github.com/Dhravya/supermemory t's a ChatGPT for your bookmarks. Import tweets or save websites and content using the chrome extension.

## other launches

- udio music https://twitter.com/udiomusic/status/1778045322654003448?t=6FDPaNxZcbSsELal6Sv7Ug
	- [comedy dialogue, sports analysis, commercials, radio broadcasts, asmr, nature sounds](https://x.com/mckaywrigley/status/1778867824217542766?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)
	- sonauto as well
- [Reka Core/Flash/Edge](https://publications.reka.ai/reka-core-tech-report.pdf)
	-   

## fundraising

- [Perplexity raised 62.7m at 1b valuation, led by dan gross](https://x.com/AravSrinivas/status/1782784338238873769)
	- https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round
- [XAI seeking 4b](https://www.bloomberg.com/news/articles/2024-04-11/elon-musk-s-xai-seeks-up-to-4-billion-to-compete-with-openai)

## Learning

- Thom Wolf - [how to train LLMs in 2024](https://youtu.be/2-SPH9hIKT8?si=wqYrDbhvgJUT2zHP)
## discussion

- soumith v fchollet https://x.com/fchollet/status/1776319511807115589
- delve is from nigeria https://twitter.com/moultano/status/1777727219097342287
	- https://x.com/jeremynguyenphd/status/1780580567215681644?s=46&t=90xQ8sGy63D2OtiaoGJuww
- [devin debunking](https://news.ycombinator.com/item?id=40008109)
	- [cognition company response](https://twitter.com/cognition_labs/status/1780661877686538448) - [engineer's response](https://twitter.com/walden_yan/status/1780014680242528406)
- [what can LLMs never do?](https://news.ycombinator.com/item?id=40179232)
- papers
	- Our 12 scaling laws (for LLM knowledge capacity)
		- prefix [low quality data with junk tokens](https://twitter.com/ZeyuanAllenZhu/status/1777513028466188404) - "when pre-training good data (e.g., Wiki) together with "junks" (e.g., Common Crawl), LLM's capacity on good data may decrease by 20x times! A simple fix: add domain tokens to your data; LLMs can auto-detect domains rich in knowledge and prioritize."

## memes

- suno memes
	- https://x.com/goodside/status/1775713487529922702
