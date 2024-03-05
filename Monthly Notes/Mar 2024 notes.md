
## openai

- Community comments
	- [GPT4 browsing buggy current search - pulls from a cache](https://x.com/AndrewCurran_/status/1764546464087159230?s=20)

## frontier models

- [Anthropic Claude 3](https://www.anthropic.com/news/claude-3-family) ([technical report](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf))
	- Haiku (small, $0.25/mtok - "available soon"), Sonnet (medium, $3/mtok - powers claude.ai, is on Amazon Bedrock and Google Vertex), Opus (large $15/mtok - powers Claude Pro)
		- **Speed**: Haiku is the fastest and most cost-effective model on the market for its intelligence category. **It can read an information and data dense research paper on arXiv (~10k tokens) with charts and graphs in less than three seconds**. Following launch, we expect to improve performance even further. [Sonnet is 2x faster than Opus and Claude 2/2.1](https://x.com/AnthropicAI/status/1764653835568726215?s=20)
		- **Vision**: The Claude 3 models have **sophisticated vision capabilities** on par with other leading models. They can process a wide range of visual formats, including photos, charts, graphs and technical diagrams.
			- [Opus can turn a 2hr video into a blogpost](https://x.com/mlpowered/status/1764718705991442622?s=20)
		- **Long context and near-perfect recall:** Claude 3 Opus not only achieved near-perfect recall, surpassing 99% accuracy, but in some cases, it even identified the limitations of the evaluation itself by recognizing that the "needle" sentence appeared to be artificially inserted into the original text by a human.
		- **Easier to use**: The Claude 3 models are better at following complex, multi-step instructions. They are particularly adept at adhering to brand voice and response guidelines, and developing customer-facing experiences our users can trust. In addition, the Claude 3 models are better at producing popular structured output in formats like JSON—making it simpler to instruct Claude for use cases like natural language classification and sentiment analysis.
	- Safety
		- Lower refusal rate - very good to combat anthropic safetyist image and topical vs gemini issues from feb
		- "Opus not only found the needle, it recognized that the inserted needle was so out of place in the haystack that this had to be an artificial test constructed by us to test its attention abilities." [from Anthropic prompt engineer](https://twitter.com/alexalbert__/status/1764722513014329620)
			- criticized by [MMitchell](https://x.com/mmitchell_ai/status/1764739357112713267?s=20) and [Connor Leahy](https://x.com/NPCollapse/status/1764740710731837516?s=20) and [Delip Rao](https://x.com/deliprao/status/1764675843542995026?s=20)
			- could be overrated - [GPT3 also does it because the needle is so out of context](https://x.com/zggyplaydguitar/status/1764791981782262103?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- Evals
		- [choosing to highlight Finance, Medicine, Philosophy domain evals rather than MMLU/HumanEval is good](https://twitter.com/DrJimFan/status/1764719012678897738)
		- [59.5% on GPQA](https://x.com/idavidrein/status/1764675668175094169?s=20) is  much better than generalist PhDs and GPT4 - GPQA author is impressed. [paper]([arxiv.org/abs/2311.12022](https://t.co/hb4u4xXzkw)).
		- [doesn't perform as well on EQbench](https://twitter.com/gblazex/status/1764762023403933864?t=6FDPaNxZcbSsELal6Sv7Ug)
	- GPT4 comparisons
		- beats GPT4 at [coding a discord bot](https://twitter.com/Teknium1/status/1764746084436607010)
		- can read/answer in ASCII
		- [fails at simple shirt drying but GPT4 doesnt](https://x.com/abacaj/status/1764698421749756317?s=20)
		- [beats GPT4 in Lindy AI evals](https://x.com/altimor/status/1764784829248262553?s=46&t=90xQ8sGy63D2OtiaoGJuww)
		- per promptbase... [probably not better than GPT4T](https://x.com/tolgabilge_/status/1764754012824314102?s=46&t=90xQ8sGy63D2OtiaoGJuww)
	- misc commentary
		- [good at d3js - can draw a self portrait](https://x.com/karinanguyen_/status/1764789887071580657?s=46&t=90xQ8sGy63D2OtiaoGJuww)
		- [200k context, can extend to 1m tokens](https://x.com/mattshumer_/status/1764657732727066914?s=20)
		- [Haiku is close to GPT4 in evals, but half the cost of GPT3.5T](https://x.com/mattshumer_/status/1764738098389225759?s=20)
		- [Trained on synthetic data](https://x.com/Justin_Halford_/status/1764677260555034844?s=20)
		- [lower loss on code is normal/unremarkable](https://twitter.com/kipperrii/status/1764673822987538622)

## Open Models

- [Together/Hazy Research Based](https://www.together.ai/blog/based) - solving the **recall-memory tradeoff** of convolutional models like Hyena/H3 in linear attention models
- [Moondream2](https://x.com/vikhyatk/status/1764793494311444599?s=20) - a small, open-source, vision language model designed to run efficiently on edge devices. Clocking in at 1.8B parameters, moondream requires less than 5GB of memory to run in 16 bit precision. This version was initialized using Phi-1.5 and SigLIP, and trained primarily on synthetic data generated by Mixtral. Code and weights are released under the Apache 2.0 license, which permits commercial use.

## other launches

- [groq launched api platform](https://x.com/atbeme/status/1764762523868508182?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- [Anthropic's API is now GA](https://x.com/mattshumer_/status/1764658247661719850?s=46&t=90xQ8sGy63D2OtiaoGJuww) (was private beta for long while)