reference [[RLHF_RLAIF]]


FLAN explainer https://overcast.fm/+_C9fgQTQE/12:05

cohere's instruction tuning
- https://docs.cohere.ai/docs/command-beta

https://simonwillison.net/2023/Mar/13/alpaca/
One of the great innovations from OpenAI was their application of [instruction tuning](https://openai.com/research/instruction-following) to GPT-3:

> To make our models safer, more helpful, and more aligned, we use an existing technique called reinforcement learning from human feedback (RLHF). On prompts submitted by our customers to the API, our labelers provide demonstrations of the desired model behavior, and rank several outputs from our models. We then use this data to fine-tune GPT-3.

Prior to this, you had to think very carefully about how to construct your prompts. Thanks to instruction tuning you can be a lot more, well, human in the way you interact with the model. “Write me a poem about pandas!” now works as a prompt, instead of “Here is a poem about pandas:”.

SEE ALSO : #### Bonus: The source of that training data? GPT-3! Here’s a fascinating detail: Those 52,000 samples they used to fine-tune the model? Those were the result of a prompt they ran against GPT-3 itself! Here’s [the prompt](https://github.com/tatsu-lab/stanford_alpaca/blob/da37bb2ecab37cae022dd07aa3ff861c446fb614/prompt.txt) they used:


https://twitter.com/zhansheng/status/1583158989889540096?s=46&t=Nd874xTjwniEuGu2d1toQQ
There are now two flavors of "instruction tuning" going around.

1. Fine-tune on instructed-formatted multi-task datasets (FLAN/T0)
2. RLHF on instructions (InstructGPT, CarperAI work)

Instruction-tune: InstructGPT-style RLHF
Instruction-finetune: Multitask+CoT tuning

> "With the InstructGPT paper we found that our models generalized to follow instructions in non-English even though we almost exclusively trained on English. We still don't know why. I wish someone would figure this out." https://twitter.com/janleike/status/1625207251630960640?s=20&t=I_xlsx5X8kNokM4XSZxIvg

## papers to read

- GPT Instruct https://openai.com/blog/instruction-following/
- Deepmind Gopher https://arxiv.org/abs/2112.11446
- [**Scaling Instruction-Finetuned Language Models (“Flan2”)**](https://arxiv.org/abs/2210.11416)***** Major Instruction Tuning work. Best open source models (Flan-T5).
- [**Multitask Prompted Training Enables Zero-Shot Task Generalization**](https://arxiv.org/abs/2110.08207) **(“T0”)** Great instruction tuning paper. While T0 is no longer SOTA, this was likely the best paper from BigScience.
- OPT-IML : Scaling Language Model Instruction Meta Learning through the Lens of Generalization
	- Instruction fine-tuning is shown (Wei et al., 2022a; Sanh et al., 2022; Chung et al., 2022a) to sig- nificantly improve the zero- and few-shot performance of large pretrained LMs (LLM). It involves fine-tuning LLMs on collections of NLP tasks using instructional style input formats.
	- There are a growing number of large meta-datasets of NLP tasks such as Super-NaturalInstructions (Wang et al., 2022), FLAN (Wei et al., 2022a) and PromptSource (Sanh et al., 2022). Recent instruction-tuning work has demonstrated success using these individual benchmarks and their com- binations (Chung et al., 2022b), with a general recommendation for scaling up the number of tasks.
	- four different instruction-tuning benchmarks: PromptSource (Sanh et al., 2022), FLAN (Wei et al., 2022a), Super-NaturalInstructions (Wang et al., 2022), and UnifiedSKG (Xie et al., 2022).
	- Recently, along similar lines as this work, Chung et al. (2022b) achieve impressive gains on the challenging of MMLU (Hendrycks et al., 2020) and Big-Bench Hard (Suzgun et al., 2022) by instruction-tuning PaLM (Chowdhery et al., 2022) and T5 (Raffel et al., 2020) on a scaled-up collection of 1.8K tasks.
- 3) [SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
	- https://twitter.com/mathemagic1an/status/1662896309588881408/photo/1
	- self instruct used in Gorilla paper
	- related - Tool LlaMA and ToolBench 
		- Large-scale instruction tuning SFT data to equip LLMs with general tool-use capability
		- https://twitter.com/TsingYoga/status/1662843257796333568?s=20
- FLAN T5
	- https://twitter.com/arankomatsuzaki/status/1583254819053047808?s=46&t=Nd874xTjwniEuGu2d1toQQ
	- https://twitter.com/quocleix/status/1583523186376785921?s=46&t=Nd874xTjwniEuGu2d1toQQ

- IBM's Dromedary, which leverages an LLM to produce a large synthetic dataset for instruction tuning:
	- https://twitter.com/generatorman_ai/status/1655941986627772419?s=20

- 