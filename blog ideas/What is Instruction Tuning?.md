reference [[RLHF]]

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