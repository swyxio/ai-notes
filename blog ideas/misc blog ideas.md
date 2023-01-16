
## FLOPS is all you need

gpt3 data - The training dataset is something like 500B tokens and not all of that is used (common crawl is processed less than once),

https://news.ycombinator.com/item?id=34400693
The human brain only consumes around 20W [1], but for numerical calculations it is massively outclassed by an ARM chip consuming a tenth of that. Conversely, digital models of neural networks need a huge power budget to get anywhere close to a brain; this estimate [2] puts training GPT-3 at about a TWh, which is about six million years' of power for a single brain. 

TWh = 10^12Wh which means a trillion-watt for 1 hour.

10^12 / 20 (power of brain) / 24 (hours in a day) / 365 (days in a year) = 5 707 762 years.

https://blog.scaleway.com/doing-ai-without-breaking-the-bank-yours-or-the-planets/
AlexNet (61m) for 1000 classifications
GPT2 (1.7b)
GPT3 (175b) took 3.14x1023 FLOPS
GPT3 took 10,000 [V100 Nvidia GPUs](https://www.nvidia.com/en-us/data-center/v100/)
V100 can handle 14 TFLOPS when using single precision
28 TFLOPS for the half-precision format
3.14x1023 -> 10k GPUs -> 13 days

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