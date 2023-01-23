
## FLOPS is all you need

https://simonwillison.net/2023/Jan/13/semantic-search-answers/
The OpenAI embedding model lets you take any string of text (up to a ~8,000 word length limit) and turn that into a list of 1,536 floating point numbers. We’ll call this list the “embedding” for the text.

These numbers are derived from a sophisticated language model. They take a vast amount of knowledge of human language and flatten that down to a list of floating point numbers—at 4 bytes per floating point number that’s 4*1,536 = 6,144 bytes per embedding—6KiB.

The distance between two embeddings represents how semantically similar the text is to each other.

varun 
- a100 - 3.3k flops
- fp8 compute - 2 petaflops
- in a 2.5 year span, nvidia has 6x compute 
- cost of training a bert model is probably down 1-2 orders of magnitude
- chinchilla - fixed compute budget, hold capability set (emergent - 20b chain of thought)- 
	- x a100 for y hours -> gives budget of flops
	- gives you n billion parameters 
	- 20n tokens
	- computing perplexity - probability of model having gnereated validation set
		- lower perplexity (negative log sum) = learning more
		- other types of proxy/eval
			- for code - perplexity doesnt matter - humaneval - but just run the code
			- not easy to validate - pubmedgpt 2.7b param -> 22% MFU, can get up to 40-50%
				- mosaic -> not that great
				- mosaic paper had another model that was 5-6x smaller 360m. 
				- google came in and fked all these guys up
- but chinchilla not end all be all - SERVING time also matters
	- param count up, model capability up, but serving slowly
	- makes sense to train a smaller model, spend more on training it
	- bigger models are more beta efficient
	- 10b param -> 5b
	- burn more on training to serve
	- larger model means higher latency
- model parallel running
	- multi nvlink - connecting multiple gpus together
	- double the memory bandwidth and flops and cost
	- latency roughly the same
- MFU - model flop utilization
	- you can lose utilization
	- what is the util of this hardware
	- for traiining MFU matters
	- but for inference - 
		- for every entire inference loop
		- you care about memory bandwidth
		- I/O bound
		- yoy, compute getting much faster compared to memory
	- 80gb a100 2TB/s, h100 3TB/s - 50% more
	- to satureate compute, need to keep things in registers
	- random reasons why you cant do batching
	- some users take on much higher latency
		- midjourney workload - minutes
			- take 30s chunk, batch as much as posisble
		- codeium - inference every keystroke
		- jasper lives in between - seconds
	- depends
		- once the output is generated - if amount is yes, low latency , fast iteration loop
- if you build on a LLM API
	- should the sharding live in the app layer or in the api layer
- read hjeff dean efficintly scaling https://www.reddit.com/r/mlscaling/comments/yrx6w6/efficiently_scaling_transformer_inference_jeff/
- 

nice chart of flops https://ourworldindata.org/brief-history-of-ai

gpt3 data - The training dataset is something like 500B tokens and not all of that is used (common crawl is processed less than once),

https://news.ycombinator.com/item?id=34400693
The human brain only consumes around 20W [1], but for numerical calculations it is massively outclassed by an ARM chip consuming a tenth of that. Conversely, digital models of neural networks need a huge power budget to get anywhere close to a brain; this estimate [2] puts training GPT-3 at about a TWh, which is about six million years' of power for a single brain. 

TWh = 10^12Wh which means a trillion-watt for 1 hour.

10^12 / 20 (power of brain) / 24 (hours in a day) / 365 (days in a year) = 5 707 762 years.

https://venturebeat.com/ai/openai-launches-an-api-to-commercialize-its-research/
GPT3 - 12m  to train, 350GB memory

https://blog.scaleway.com/doing-ai-without-breaking-the-bank-yours-or-the-planets/
AlexNet (61m) for 1000 classifications
GPT2 (1.7b)
GPT3 (175b) took 3.14x1023 FLOPS
GPT3 took 10,000 [V100 Nvidia GPUs](https://www.nvidia.com/en-us/data-center/v100/)
V100 can handle 14 TFLOPS when using single precision
28 TFLOPS for the half-precision format
3.14x1023 -> 10k GPUs -> 13 days

https://info.deeplearning.ai/generated-code-makes-overconfident-programmers-chinas-autonomous-drone-carrier-does-bot-therapy-require-informed-consent-mining-for-green-tech-1?ecid=ACsprvuGDgDT6KPrs7TEv1tHiMmOJ3ZifowbdON1zkA-JQ4G4-Nk-3DgGfZMBYUSIMQxgzVFMReI
_ChatGPT’s predecessor GPT-3 has 175 billion parameters. Using 16-bit, floating-point bytes, it would take around 350GB to store its parameters (many reports say 800GB)_
_In comparison, Wikipedia occupies about 150GB (50GB for text, 100GB for images)_ _While the comparison is far from apples to apples, the fact that an LLM has more memory than is needed to store Wikipedia suggests its potential to store knowledge._
_Wikipedia contains a minuscule fraction of the knowledge available on the internet, which by some estimates amounts to 5 billion GB._

 pubmedgpt story https://overcast.fm/+Jy_x31W5I/10:00 50b tokens of data - all published journals

full training - 10x-100x of a single run

energy
- 1 V100 GPU - 300W per day
- 10k V100 GPUs x 13 days = 936 MWh
- Power Usage Effectiveness - 1.5 datacenter (as low as 1.15)
	- so true cost 936 * 1.5 = 1404 MWh
- 1 MW = 2k homes
- another GPT3 estimate 190,000 kWh
	- https://www.theregister.com/2020/11/04/gpt3_carbon_footprint_estimate/
	- 85,000 kg of CO2 equivalents, the same amount produced by a new car in Europe driving 700,000 km, or 435,000 miles, which is about twice the distance between Earth and the Moon



## perplexity

ai cheating detection
perplexity with gptzero



## waves in AI products

- text
	- first it was headlines
	- then it was copywriting/product description
	- then its fiction
- first it was architecture
- then it was dreambooth pfps
- then it was settings (levelsio)
- then it was products
	- Pebbley -  inpainting for product placement https://twitter.com/alfred_lua/status/1610641101265981440?s=46&t=RMPT1jJedELVkL2Aby-40g
	- Flair AI https://twitter.com/mickeyxfriedman/status/1613251965634465792
	- https://www.stylized.ai/

## best ai newsletters

- andrew ng https://www.deeplearning.ai/the-batch/


## language models as wizard of oz 

languagemodels help you do things that dont scale
 youtu.be/4RMjQal_c4U
- prototyping
- product specs -> product
	- api specs are easier
- redux thing
- development workflow
	- PRD
	- api prototype
	- frontend prototype?
	- generate code and tests
	- tweak code
	- commit and deploy

## transofmrers are eating the world

- karpathy observation
- tayml model architecture chart



## simple ai projects to start

- reproduce the causality benchmark https://github.com/amit-sharma/chatgpt-causality-pairs
- make up your own tests?
	- LSAT, Chemistry, real estate, medical boards...
- make your own search engine
	- https://twitter.com/rileytomasek/status/1603854647575384067?s=20


## what's needed in 2023

https://twitter.com/saranormous/status/1601388294461218821?s=20
- table stakes: shallow integration of generic models available through limitied APIs
- next:
	- Focusing on driving down the cost of pretraining, ongoing training and inference
	- **Personalizing with conditioning or mass fine-tuning**
	- **Inventing creative interfaces**
	- Managing hallucinations and "reasonable but incorrect" answers
	- Shaping the data that we have into the data models need
	- **Exploring the unintuitive ways models are superhuman** (moravec's paradox)
	- Incorporating planning for more sophisticated, multi-step tasks
	- Building a "foundational model for the real world" (and thus robotics)
	- Crossing valleys of "cool demo but unusable in prod"


## why radiologists didnt go away

https://twitter.com/bengoldhaber/status/1611074716927922177?s=46&t=fAgqJB7GXbFmnqQPe7ss6w


### prompt engineering techniques

- structured prompting https://twitter.com/mathemagic1an/status/1604802787296284674/photo/1 - breaking context limits
	- Get 1000s of in-context samples => split them into M groups, each small enough to fit in regular context length => encode each of M groups using LLM encoder => combine these encoded groups and attend over a scaled version of the combination simultaneously
	- Traditional attention mechanisms scale quadratically (O(N^2)) in memory/time complexity with the number of in-context samples This scales O(N^2/M)

### interviews

- dust.tt
- goodside
- sharif shameem
- orchard.ink guys