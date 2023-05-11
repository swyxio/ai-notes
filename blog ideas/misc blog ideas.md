
## alibi and the context of infinity

gpt4 demo
anthropic https://news.ycombinator.com/item?id=35904773

## data augmented retrieval

llamaindex
vs 
langchain

approaches

### ways to augment foundation models

- RLHF/Finetuning (ylecun cake analogy)
	- focus model
- stuff examples in context
	- search similar knowledge and hallucinate/summarize
- [ReACT](https://www.geoffreylitt.com/2023/01/29/fun-with-compositional-llms-querying-basketball-stats-with-gpt-3-statmuse-langchain.html) 
	- augment capability to get knowledge
	- fixie, zapier

### illustrating  langchain concepts

- https://blog.langchain.dev/agent-toolkits/
	- By agents we mean a system that uses an LLM to decide what actions to take in a repeated manner, where future decisions are made based on observing the outcome of previous actions.


## on the Bitter lesson

- https://archive.is/FEGf1 In an influential 2019 [essay](https://archive.is/o/FEGf1/https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf), the A.I. researcher Richard Sutton explains that early speech-recognition programs were loaded with specialized linguistics knowledge—not just about syntax, grammar, and phonetics but about how the shape of the human mouth constrained what sounds were possible. Despite their sophistication, these programs didn’t work very well. In the nineteen-seventies, there was a turn toward statistical methods, which dropped expert knowledge in favor of patterns learned from data—for instance, about which sounds and words tended to go together. The success of that approach bled out into the rest of A.I., leading the field to center much of its effort on statistics drawn from huge amounts of data.
	- response https://staff.fnwi.uva.nl/m.welling/wp-content/uploads/Model-versus-Data-AI-1.pdf


### SCALE IS NOT all you need

we dont know what optimal param size for all these models
https://overcast.fm/+34A2VOlx0/11:51

## FLOPS is all you need


https://simonwillison.net/2023/Jan/13/semantic-search-answers/
The OpenAI embedding model lets you take any string of text (up to a ~8,000 word length limit) and turn that into a list of 1,536 floating point numbers. We’ll call this list the “embedding” for the text.

These numbers are derived from a sophisticated language model. They take a vast amount of knowledge of human language and flatten that down to a list of floating point numbers—at 4 bytes per floating point number that’s 4*1,536 = 6,144 bytes per embedding—6KiB.

The distance between two embeddings represents how semantically similar the text is to each other.

https://huggingface.co/blog/large-language-models
Researchers estimate that the human brain contains an average of [86 billion neurons](https://pubmed.ncbi.nlm.nih.gov/19226510/) and 100 trillion synapses.

varun 
- a100 - 3.3k flops
- fp8 compute - 2 petaflops
- exascale 
	- https://www.theguardian.com/technology/2023/mar/15/uk-to-invest-900m-in-supercomputer-in-bid-to-build-own-britgpt?utm_source=substack&utm_medium=email
	- An exascale computer is one that can carry out more than one billion billion simple calculations a second, a metric known as an “exaflops”. Only one such machine is known to exist, Frontier, which is housed at America’s Oak Ridge National Laboratory and used for scientific research – although supercomputers have such important military applications that it may be the case that others already exist but are not acknowledged by their owners. Frontier, which cost about £500m to produce and came online in 2022, is more than twice as powerful as the next fastest machine.
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
- read jeff dean efficintly scaling https://www.reddit.com/r/mlscaling/comments/yrx6w6/efficiently_scaling_transformer_inference_jeff/
- 

nice chart of flops https://ourworldindata.org/brief-history-of-ai


compression
- stable diffusion - 100k gb compressed to 1.2gb https://www.listennotes.com/podcasts/the-logan-bartlett/ep-46-stability-ai-ceo-emad-8PQIYcR3r2i/ 21mins in
- https://en.wikipedia.org/wiki/Hutter_Prize


stable diffusion costs
- https://overcast.fm/+34A2VOlx0/32:33
- sd1 took 200,000 A100 hours to... Like 600k retail
- openclip 5m
- We had a machine learning ops team of four people managing 4,000 A100s. Now we're up to nearly 6,000. 

"Chinchilla's Wild Implications" [https://lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications](https://t.co/fgyyHePwZ8)
- to train PaLM 540B optimally you need 6.7T tokens (5x chinchilla)
- https://twitter.com/ethanCaballero/status/1554153812432240643?s=20

gpt3 data - The training dataset is something like 500B tokens and not all of that is used (common crawl is processed less than once),

-   _In a typical year, a child might hear around 10 million words (with huge variance depending on factors such as the [family](https://info.deeplearning.ai/e3t/Ctc/LX+113/cJhC404/VVYfmW6XqStfW8MB_k715LCG0W1YzZHS4Ww_v6N6R-BYJ3q3n5V1-WJV7CgQCzW4MTNCW3NPYFqW5ppvZn8V7692W6WkNRS3NlQ9lW4dZmTp4jZSq7Vhk95K84Lg6BN7MpShfdK4_ZW2lwp3g7Y1VNWW8cchlP7Rj_8ZW5chJpG2lmjKbW2ZclWC78xFmFW9lWrpl7HgWNCW3nGmq347VF3MW1B4S4w26333FW2z8kLr5K73zCW3h5QSK7V1_3ZV7hxSd2C3vXsW2FStmG2sW5k_W7Cnmdv7RrX7lW78hWnh4yDlFPW6Q7jKW4ffpx93h5M1)). So, by age 10, the child might have heard 100 million words._ 
-   _If you read 24/7 for a year at a rate of 250 words per minute, you’d read about 130 million words annually._ 
-   _GPT-3 was trained on about 500,000 million words._ _An individual human would need to spend dozens of lifetimes reading 24/7 to see the number of words that GPT-3 considered during its training._



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

HOWEVER: [andrew ng](https://info.deeplearning.ai/chatbots-gone-wild-surveillance-takes-hold-rules-for-military-ai-robot-training-streamlined-1?ecid=ACsprvsVL700ZN_pWRpntYYV2Kh_bE6vqj7mNADQhAPxScJEAqpVKXg-jUPmrVoLa9-ohOKtM01N&utm_campaign=The%20Batch&utm_medium=email&_hsmi=247273528&_hsenc=p2ANqtz-_UTsa6nrxOWblO1ikYvjowBI8PuiXWGXN-kC9yNuF2iObAYoqjnbilIVOBHBx09gR9py6t1YUOWaGBWPFHtwhQcEFTKQ&utm_content=247274434&utm_source=hs_email) says _I don’t see a realistic path to getting an LLM with a fixed set of parameters to both (i) demonstrate deep and broad knowledge about the world and (ii) be accurate most of the time. A 175B-parameter model simply doesn’t have enough memory to know that much._

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
	- https://www.deeplearning.ai/the-batch/issue-192/ In 2022, training BLOOM emitted as much carbon dioxide as an average human would in four years. In 2020, training GPT-3, which is around the same size, emitted more than an average human would in 91 years.


Rocket league
- -   Nexto learned by playing against itself in approximately [250,000 hours](https://info.deeplearning.ai/e3t/Ctc/LX+113/cJhC404/VVYfmW6XqStfW8MB_k715LCG0W1YzZHS4Ww_v6N6R-BYJ3q3n5V1-WJV7CgBTzW12FgKD4y5jKJW1c9JKy5-wF5KW8S9jJt8gxZQvW1fBQKj8MyXrYW3ZhCr16cxgXCVqSbb118MRgFW2MTQmV8zPZWxN85N7SS65FvqN3MHvC0cZzBbW6g3PWw3sGyt7W5HJdr85fVPKPVLRgrn4XHSPXN70rNNDdq0gKW3Tf1gc3g5ybZW90Mnrq8_PXJ9W63drKp5zxGZ7W6SHd4m5SXSMRW5qWB4t1qbGDbW5xp3z_4vZSXMN8tLpP5wgw203cXJ1) (roughly 29 years 24/7) worth of gameplay, typically playing many accelerated games simultaneously. The developers estimate that its performance matches that of the top 1 percent of players. https://www.youtube.com/watch?v=jQHt2O0PkCQ&t=360s


## perplexity

ai cheating detection
perplexity with gptzero
- https://github.com/BurhanUlTayyab/GPTZero
detectgpt
https://twitter.com/chelseabfinn/status/1618822672489779201?s=46&t=REbUC7rjX6u_QSXYIzeDLw


https://twitter.com/_eric_mitchell_/status/1618820358043475969?s=46&t=REbUC7rjX6u_QSXYIzeDLw

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
	- booth ai https://twitter.com/nick_locascio?s=21&t=0HHr1d45XGJ24amPDvbz5w

## best ai newsletters

- andrew ng https://www.deeplearning.ai/the-batch/


## language models as wizard of oz 


- https://spindas.dreamwidth.org/4207.html redux prototypers
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
- sahil lavignia book author question
	- https://github.com/slavingia/askmybook
	- https://m.youtube.com/watch?v=V3RTA9ZbEPw&feature=youtu.be
- train lang model from scratch
	- https://huggingface.co/blog/how-to-train
- make same.energy  - vision transformer with vector database
	- https://news.ycombinator.com/item?id=34614449
- integrate all the tools https://www.samdickie.me/writing/experiment-1-creating-a-landing-page-using-ai-tools-no-code
- shovel ready projects https://docs.google.com/document/d/16tIpuu8WexxlM-3XpEub05BOTCFNiyvIVzZNXX0uDWw/edit# from ivan https://www.vendrov.ai/
- make your own controlnet image https://news.ycombinator.com/item?id=35112641
	- 1. I used a ControlNet Colab from here based on SD 1.5 and the original ControlNet app: [https://github.com/camenduru/controlnet-colab](https://github.com/camenduru/controlnet-colab)
	- Screenshotted a B/W OpenAI logo from their website.
	- Used the Canny adapter and the prompt: charcuterie board, professional food photography, 8k hdr, delicious and vibrant


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

https://twitter.com/_jasonwei/status/1636436324139028481?s=20



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