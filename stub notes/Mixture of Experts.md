
huggingface reading
-  [What is a Mixture of Experts?](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe)
-   [A Brief History of MoEs](https://huggingface.co/blog/moe#a-brief-history-of-moes)
-   [What is Sparsity?](https://huggingface.co/blog/moe#what-is-sparsity)
-   [Load Balancing tokens for MoEs](https://huggingface.co/blog/moe#load-balancing-tokens-for-moes)
-   [MoEs and Transformers](https://huggingface.co/blog/moe#moes-and-transformers)
-   [Switch Transformers](https://huggingface.co/blog/moe#switch-transformers)
-   [Stabilizing training with router Z-loss](https://huggingface.co/blog/moe#stabilizing-training-with-router-z-loss)
-   [What does an expert learn?](https://huggingface.co/blog/moe#what-does-an-expert-learn)
-   [How does scaling the number of experts impact pretraining?](https://huggingface.co/blog/moe#how-does-scaling-the-number-of-experts-impact-pretraining)
-   [Fine-tuning MoEs](https://huggingface.co/blog/moe#fine-tuning-moes)
-   [When to use sparse MoEs vs dense models?](https://huggingface.co/blog/moe#when-to-use-sparse-moes-vs-dense-models)
-   [Making MoEs go brrr](https://huggingface.co/blog/moe#making-moes-go-brrr)
    -   [Expert Parallelism](https://huggingface.co/blog/moe#parallelism)
    -   [Capacity Factor and Communication costs](https://huggingface.co/blog/moe#capacity-factor-and-communication-costs)
    -   [Serving Techniques](https://huggingface.co/blog/moe#serving-techniques)
    -   [Efficient Training](https://huggingface.co/blog/moe#more-on-efficient-training)
-   [Open Source MoEs](https://huggingface.co/blog/moe#open-source-moes)
-   [Exciting directions of work](https://huggingface.co/blog/moe#exciting-directions-of-work)
-   [Some resources](https://huggingface.co/blog/moe#some-resources)

HuggingGPT, JARVIS, ToolFormer, TaskMatrix, Chameleon.
https://www.reddit.com/r/MachineLearning/comments/137rxgw/comment/jiuof1w/?utm_source=reddit&utm_medium=web2x&context=3


Google Switch C 1.6T MoE model
	- https://news.ycombinator.com/item?id=38352794

Switch Transformers
- https://twitter.com/TheTuringPost/status/1670793833964097537?s=20



https://twitter.com/amanrsanger/status/1690072804161650690?s=20
- MOE allows models to scale parameter count and performance without scaling inference or training costs. This means I could serve an MOE model significantly faster and cheaper than a quality-equivalent dense model [1]. (2/7)
- Why is this bad for on-device inference? On-device inference is extremely memory limited. Apple’s M2 Mac has just 24GB of GPU RAM. Even with 4-bit quantization we can barely fit a 48B param model. And that model would see latency of <6 tok/s [2] (3/7)
- An A100 has 80GB of memory. We can serve a quality-equivalent MOE model with 100B parameters, taking up 50GB of RAM. It would likely cost around $0.2/1M generated token and less than $0.1/1M tokens running at 26 tok/s. If just maximizing speed, we could hit 173 tok/s! [3] (4/7)
- You can also serve massive MOE models splitting model weights across GPUs. On a single machine, using 640 GB of GPU RAM, you can easily serve >1T parameter MOE model (i.e. near GPT-4 level) with 4-bit quant. And it would cost within a factor of two of serving GPT-3

## gpt4 MoE rumors

- [Non-determinism in GPT-4 is caused by Sparse MoE](https://152334h.github.io/blog/non-determinism-in-gpt-4/)