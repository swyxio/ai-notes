
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