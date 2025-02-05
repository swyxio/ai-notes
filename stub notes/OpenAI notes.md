
- origins and recruiting researchers - 2016 https://www.wired.com/2016/04/openai-elon-musk-sam-altman-plan-to-set-artificial-intelligence-free/
	- decent high level history of openai
- 2015-2016 - had 3 original objectives https://www.youtube.com/watch?v=xXCBz_8hM9w
  	- deep learning/big unsupervised training run
  	- solve RL
	- <120 people
- bob mcgrew recounting https://www.youtube.com/watch?v=eW7rUtYHD9U
- openai 2017 501c3 tax disclosures with key members and original founding statement https://regmedia.co.uk/2019/05/02/openai_tax_2017.pdf
	- 2016 http://990s.foundationcenter.org/990_pdf_archive/810/810861541/810861541_201612_990.pdf
	- news https://www.nytimes.com/2018/04/19/technology/artificial-intelligence-salaries-openai.html
- openai policy thinking with jack clark and amanda askell - 2019 https://share.transistor.fm/s/dfbe0ca7
- GPT2 and leadership team - 2020 https://www.technologyreview.com/2020/02/17/844721/ai-openai-moonshot-elon-musk-sam-altman-greg-brockman-messy-secretive-reality/
- 2021 openai fund https://x.com/gdb/status/1397610307699281921?s=20
- openai-anthropic split was over the GPT3 api https://archive.is/YTOeD#selection-333.226-337.82
- openai rebrand and identity https://twitter.com/asselinpaul/status/1636938731398787074
- elon and sama split - 2023 https://www.semafor.com/article/03/24/2023/the-secret-history-of-elon-musk-sam-altman-and-openai
- july 2022-march 2023 shipping velocity https://twitter.com/E0M/status/1635727471747407872
- 2023
	- feb 2023 - openai foundry notes https://twitter.com/labenz/status/1630284912853917697
	- march 2023 - turned off codex and then changed their minds  https://twitter.com/deepfates/status/1638212305887567873?s=20
	- apr 2023 - valued at 27b with 400 employees internally https://twitter.com/frantzfries/status/1647976195693182978?s=46&t=90xQ8sGy63D2OtiaoGJuww
	- may 2023 - losing 500m a year? https://twitter.com/amir/status/1654218677074681858?s=20
	- may - sama and greg brockman podcast with reid hoffman https://fortune.com/2023/05/04/openai-success-chatgpt-business-technology-rule-greg-brockman/
	- June - functions
	- july - code interpreter
	- aug - finetuning
	- aug - superalignment
		- jan leike https://axrpodcast.libsyn.com/24-superalignment-with-jan-leike
	- oct 2023 - [openai explains what they do for global affairs people](https://openai.com/global-affairs/openai-technology-explained)
- leadup to ChatGPT
	- https://www.theverge.com/2024/12/12/24318650/chatgpt-openai-history-two-year-anniversary
 		- both [microsoft](https://x.com/christinahkim/status/1869159721003081825) and anthropic had similar chatbots but didnt launch 

## OpenAI deprecation and models

- https://twitter.com/OfficialLoganK/status/1677682265311043584
	- have to switch models technically every 6 months (we release a new model and then the replacement comes out ~3 months later then the 3 month deprecation window starts).

## ilya sutskever

- 2 core ideas behind openai https://twitter.com/radekosmulski/status/1639123080646897664
	- Unsupervised learning through compression. Turns out, if you want to compress a lot of data, you have to discover the secrets that live in it.
	- Reinforcement learning is impt. But the RL from playing Dota merged with Transformers to give us Reinforcement Learning from Human Feedback. 
	- Bigger is better. predict text + scale => world model
	- highlights from https://youtu.be/GI4Tpi48DlA?si=u7guXWVvIS5OsRaH
- [Ilya Sutskever on No Priors pod](https://www.youtube.com/watch?v=Ft0gTO2K85A

## greg brockman

- https://blog.gregbrockman.com/my-path-to-openai
- https://blog.gregbrockman.com/how-i-became-a-machine-learning-practitioner
- https://blog.gregbrockman.com/its-time-to-become-an-ml-engineer
- his rescuetime https://twitter.com/sama/status/792898456650076160/photo/1 
	- 2021 https://twitter.com/gdb/status/1359603222491451394

- at mit conference with sal khan
	- identifies as a self taught programmer
	- after stripe considered  going into VR and programming education inspired by sal khan
	- inspired by alan turing turjng test paper https://academic.oup.com/mind/article/LIX/236/433/986238
		- said you need rewards and punishment
		- machine can learn things that i dont even know
## sama

- intentionally did not give chatgpt a human name - want to distance from “person-ness” but accept that others will do it https://overcast.fm/+SusunrACk/12:51
- [not fired by YC](https://news.ycombinator.com/item?id=40521657)

## mainstream puff pieces

- https://www.wired.com/story/what-openai-really-wants/
	- OpenAI’s road to relevance really started with its hire of an as-yet-unheralded researcher named Alec Radford, who joined in 2016, leaving the small Boston AI company he’d cofounded in his dorm room. After accepting OpenAI’s offer, he told his high school alumni magazine that taking this new role was “kind of similar to joining a graduate program”—an open-ended, low-pressure perch to research AI.
	- The role he would actually play was more like Larry Page inventing PageRank.
	- Radford, who is press-shy and hasn’t given interviews on his work, responds to my questions about his early days at OpenAI via a long email exchange. His biggest interest was in getting neural nets to interact with humans in lucid conversation. This was a departure from the traditional scripted model of making a chatbot, an approach used in everything from the primitive ELIZA to the popular assistants Siri and Alexa—all of which kind of sucked. “The goal was to see if there was any task, any setting, any domain, any _anything_ that language models could be useful for,” he writes. At the time, he explains, “language models were seen as novelty toys that could only generate a sentence that made sense once in a while, and only then if you really squinted.” His first experiment involved scanning 2 billion Reddit comments to train a language model. Like a lot of OpenAI’s early experiments, it flopped. No matter. The 23-year-old had permission to keep going, to fail again. “We were just like, Alec is great, let him do his thing,” says Brockman.

His next major experiment was shaped by OpenAI’s limitations of computer power, a constraint that led him to experiment on a smaller data set that focused on a single domain—Amazon product reviews. A researcher had gathered about 100 million of those. Radford trained a language model to simply predict the next character in generating a user review.
