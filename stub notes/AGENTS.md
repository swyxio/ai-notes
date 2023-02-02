
- OpenAI WebGPT https://openai.com/blog/webgpt/
	- The variant is called WebGPT and can ask search queries, follow links, scroll up and down web pages, and prove the sources of the answers it finds.
	- 3rd party clone https://cut-hardhat-23a.notion.site/code-for-webGPT-44485e5c97bd403ba4e1c2d5197af71d
- Dust XP1  (WebGPT [clone](https://twitter.com/dust4ai/status/1587104029712203778) chrome extension)
	- announcement https://twitter.com/spolu/status/1602692992191676416
	- https://xp1.dust.tt/
	- **XP1** is an assistant based on GPT (text-davinci-003) with access to your browser tabs content. It is geared (prompted) towards productivity and can be used to help you with your daily tasks (such as answering emails, summarizing documents, extracting structured data from unstructured text, ...)
- Adept ACT-1  https://www.adept.ai/act
	- ACT-1 is a large-scale Transformer trained to use digital tools — among other things, we recently taught it how to use a web browser. Right now, it’s hooked up to a Chrome extension which allows ACT-1 to observe what’s happening in the browser and take certain actions, like clicking, typing, and scrolling, etc. The observation is a custom “rendering” of the browser viewport that’s meant to generalize across websites, and the action space is the UI elements available on the page.
	- Features demoed:
		- human request -> execute it over long time horizon including observations on website
		- 10 clicks in salesforce -> 1 sentence
		- make execl formula in google sheets
		- coachable with human feedback
	- clone: ACTGPT https://yihui.dev/actgpt
	- clone https://twitter.com/divgarg9/status/1619073088192417792?s=46&t=PuOBK71y8IUBOdSULtaskA
- MineDojo, a new open framework to help the community develop generally capable agents. 
	- MineDojo features a simulator suite based on Minecraft, a massive internet database (YouTube, Wiki, Reddit), and a simple but promising foundation model recipe for agents. https://twitter.com/DrJimFan/status/1595459499732926464
	  - We argue that there are 3 main ingredients for generalist agents to emerge. 
	  - First, an open-ended environment that allows an unlimited variety of tasks and goals. Earth is one example, as it is rich enough to forge an ever-expanding tree of life forms and behaviors. What else?
	  - Second, a large-scale knowledge base that teaches an AI not only *how* to do things, but also *what* are the useful things to do. GPT-3 learns from web text alone, but can we give our agent much richer data, such as video walkthroughs, multimedia tutorials, and free-form wiki?
	  - Third, an agent architecture flexible enough to pursue any task in open-ended environments, and scalable enough to convert large-scale, multimodal knowledge sources into actionable insights. Something like an *embodied* GPT-3.
		👉Website: https://minedojo.org
		👉NeurIPS: https://neurips.cc/virtual/2022/poster/55737
		👉Arxiv: https://arxiv.org/abs/2206.08853
		👉Code, models, tools: https://github.com/MineDojo
- Deepmind DreamerV3: an RL agent that generalizes across domains without human/expert input! 
	- https://twitter.com/mathemagic1an/status/1613300360789262340
	- it solves the Minecraft Diamond challenge without human data. https://twitter.com/danijarh/status/1613161946223677441
	- The key contribution of DreamerV3 is an algorithm that works out of the box on new application domains, without having to adjust hyperparameters. This reduces the need for expert knowledge and computational resources, making reinforcement learning broadly applicable.
	- They train 3 separate models that work together: - world model: predicts future outcomes of actions - critic: judges the value of situations - actor: learns to reach valuable situations
	- compare to previous work Gato, we are seeing exciting advancements in RL. (Gato learned to do over 600 different tasks in the *same set of parameters*, although it used behavior modeling - not pure RL - to get there.) https://deepmind.com/publications/a-generalist-agent 
- Langchain Agent
	- https://twitter.com/nickscamara_/status/1614119040263360512?s=20
	- https://langchain.readthedocs.io/en/latest/modules/agents/implementations/mrkl.html


- Whatsapp + ChatGPT
  - https://twitter.com/danielgross/status/1598735800497119232
  - telegram port https://twitter.com/altryne/status/1598822052760195072
  - "unofficial api" 
    - https://twitter.com/taranjeetio/status/1598747137625333761?s=20
    - https://twitter.com/transitive_bs/status/1598841144896417794  

https://twitter.com/realGeorgeHotz/status/1591181077292679168
https://twitter.com/realbrendanb/status/1593414620928413696
https://twitter.com/samipddd/status/1620887558136946689?s=20
https://twitter.com/ai__pub/status/1593365765012434944
	- https://blog.swift.vc/the-next-generation-of-large-language-models-will-blow-your-mind-and-disrupt-your-business-3b913d6dfa8a

thread on GPT3 agents https://twitter.com/GrantSlatton/status/1600890243452137472?s=20

https://jmcdonnell.substack.com/p/the-near-future-of-ai-is-action-driven?sd=pf # The Near Future of AI is Action-Driven …and it will look a lot like AGI