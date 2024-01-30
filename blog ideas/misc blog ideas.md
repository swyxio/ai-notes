

## ai waifu/gf companies

- callannie.ai
- chai-research.com (chai ai)
- https://www.moemate.io/?searchPage=1&public=1
- check openrouter rankings




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



### prompt engineering techniques

- structured prompting https://twitter.com/mathemagic1an/status/1604802787296284674/photo/1 - breaking context limits
	- Get 1000s of in-context samples => split them into M groups, each small enough to fit in regular context length => encode each of M groups using LLM encoder => combine these encoded groups and attend over a scaled version of the combination simultaneously
	- Traditional attention mechanisms scale quadratically (O(N^2)) in memory/time complexity with the number of in-context samples This scales O(N^2/M)
