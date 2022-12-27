<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [Infrastructure](#infrastructure)
- [Optimization](#optimization)
- [hardware issues](#hardware-issues)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Infrastructure

- bananadev cold boot problem https://twitter.com/erikdunteman/status/1584992679330426880?s=20&t=eUFvLqU_v10NTu65H8QMbg
- replicate.com
- banana.dev
- huggingface.co
- lambdalabs.com
- astriaAI
- SkyPilot
	- https://medium.com/@zongheng_yang/skypilot-ml-and-data-science-on-any-cloud-with-massive-cost-savings-244189cc7c0f
- cost of chatgpt - https://twitter.com/tomgoldsteincs/status/1600196981955100694
	- A 3-billion parameter model can generate a token in about 6ms on an A100 GPU
	- a 175b param it should take 350ms secs for an A100 GPU to print out a single word
	- You would need 5 80Gb A100 GPUs just to load the model and text. ChatGPT cranks out about 15-20 words per second. If it uses A100s, that could be done on an 8-GPU server (a likely choice on Azure cloud)
	- On Azure cloud, each A100 card costs about $3 an hour. That's $0.0003 per word generated.
	- The model usually responds to my queries with ~30 words, which adds up to about 1 cent per query.
	- If an average user has made 10 queries per day, I think it’s reasonable to estimate that ChatGPT serves ~10M queries per day.
	- I estimate the cost of running ChatGPT is $100K per day, or $3M per month.

stack example
- https://twitter.com/ramsri_goutham/status/1604763395798204416?s=20
	- Here is how we bootstrapped 3 AI startups with positive unit economics - 
	1. Development - Google Colab 
	2. Inference - serverless GPU providers (Tiyaro .ai, modal .com and nlpcloud)
	3. AI Backend logic - AWS Lambdas 
	4. Semantic Search - Free to start vector DBs (eg: pinecone .io) 
	5. Deployment - Vercel + Supabase


## Optimization

- For single-GPU performance, there are 3 main areas your model might be bottlenecked by. Those are: 1. Compute, 2. Memory-Bandwidth, and 3. Overhead. Correspondingly, the optimizations that matter *also* depend on which regime you're in. https://horace.io/brrr_intro.html ([tweet](https://twitter.com/cHHillee/status/1503803015941160961))
	- [![https://pbs.twimg.com/media/FjkqgJ8VUAAN4oA?format=jpg&name=medium](https://pbs.twimg.com/media/FjkqgJ8VUAAN4oA?format=jpg&name=medium)](https://twitter.com/cHHillee/status/1601371646756933632?s=20)
	- RELATED hardware influencing pytorch design - compute-bound https://twitter.com/cHHillee/status/1601371638913638402?s=20
- bootstrapping data 
	- "Data engine" - use GPT3 to generate 82k samples for instruction tuning - generates its own set of new tasks, outperforms original GPT3   https://twitter.com/mathemagic1an/status/1607384423942742019
	- "[LLMs are Reasoning Teachers](https://arxiv.org/abs/2212.10071)"
		- https://twitter.com/itsnamgyu/status/1605516353439354880?s=20
		- We propose Fine-tune-CoT: fine-tune a student model with teacher-generated CoT reasoning, inspired by Zero-shot CoT
		- All of our experiments use public APIs from OpenAI on a moderate budget of just $50-200 per task. The code is already on GitHub


## hardware issues

- https://hardwarelottery.github.io ML will run into an asymptote because matrix multiplication and full forward/backprop passes are ridiculously expensive. What hardware improvements do we need to enable new architectures?
- "bitter lessons" - http://incompleteideas.net/IncIdeas/BitterLesson.html https://twitter.com/drjwrae/status/1601044625447301120?s=20
- related: https://www.wired.com/2017/04/building-ai-chip-saved-google-building-dozen-new-data-centers/
- transformers won because they were more scalable https://arxiv.org/pdf/2010.11929.pdf


## cost trends - wright's law

- We believe the cost to train a neural net will fall 2.5x per year through 2030. AND we expect budgets to continue to balloon, doubling annually at least through 2025. Combine the two: Neural net capability should increase by ~5,000x by 2025
- https://twitter.com/wintonARK/status/1557768036169314304?s=20