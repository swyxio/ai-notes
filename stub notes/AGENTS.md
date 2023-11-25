## timeline

- 1990-2010s schmidhuber work https://people.idsia.ch/~juergen/lecun-rehash-1990-2022.html https://twitter.com/SchmidhuberAI/status/1683870175299239937
- adept fundraising
- webgpt
- nov 2022 - [harrison included agents](https://twitter.com/hwchase17/status/1595456660507459585)
- tool use - react, toolformer, chatgpt plugins

## Survey articles

- https://www.sequoiacap.com/article/autonomous-agents-perspective/

## React Model

- https://react-lm.github.io and [google blog](https://ai.googleblog.com/2022/11/react-synergizing-reasoning-and-acting.html)
	- https://interconnected.org/home/2023/03/16/singularity
	- https://blog.langchain.dev/agent-toolkits/
	- https://www.geoffreylitt.com/2023/01/29/fun-with-compositional-llms-querying-basketball-stats-with-gpt-3-statmuse-langchain.html
- Automatic Reasoning & Tool-Use of LLMs
	- Retrieves examples of reasoning & tool use from task library -Writes its own program, pauses when tool call is encountered, integrates output -Much better than few-shot prompting [tweet](https://mobile.twitter.com/johnjnay/status/1636555772082499584?s=12&t=90xQ8sGy63D2OtiaoGJuww)
- alternative
	- Karpas, E., Abend, O., Belinkov, Y., Lenz, B., Lieber, O., Ratner, N., . . ., Tenenholtz, M., MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning, 2022.

## other agent paradigms

https://www.geeksforgeeks.org/agents-artificial-intelligence/#

Reflexion: an autonomous agent with dynamic memory and self-reflection https://arxiv.org/abs/2303.11366 https://twitter.com/johnjnay/status/1638696539257184258
A Self-Reflecting LLM Agent

Equips LLM-based agent w/ 
-dynamic memory
-a self-reflective LLM
-a method for detecting hallucinations
self critique works well to fix first shot problems 
- https://twitter.com/ericjang11/status/1639882111338573824

browser agents
- hyperwrite personal assistant https://venturebeat.com/ai/hyperwrite-unveils-breakthrough-ai-agent-that-can-surf-the-web-like-a-human/

https://twitter.com/jh_damm/status/1646233661832929280?s=20
  
React Style Prompting 
BabyAGI: task planning & prioritization for endless iteration 
Agent Duos: collaborative problem-solving 
Stacking Agents: task delegation & collaboration


### Roleplaying / Multi-Agent systems
- camel and autogen
	- https://arxiv.org/pdf/2308.08155v1.pdf 
	- https://github.com/camel-ai/camel
- metagpt
- Multi-Agent Systems: • 
	- BabyAGI: BabyAGI [6] is an example implementation of an AI-powered task management system in a Python script (according to its own documentation). In this implemented system, multiple LLM-based agents are used. For example, there is an agent for creating new tasks based on the objective and the result of the previous task, an agent for prioritizing the task list, and an agent for completing tasks/sub-tasks. BabyAGI is a multi-agent system with a static agent communication pattern, i.e., a predefined order of agent communication. • 
	- CAMEL: CAMEL [28] is a communicative agent framework. It demonstrates how role-playing can be used to let chat agents communicate with each other for task completion. It also records agent conversations for behavior analysis and capability understanding. CAMEL does not support tool-using, such as code execution. Inception prompting technique is used to achieve autonomous cooperation between agents. • 
	- MetaGPT: MetaGPT [20] is a multi-agent framework for assigning different roles to GPTs to form a collaborative software entity for complex tasks. It is a specialized LLM-based multi-agent framework for collaborative software development. • 
	- Multi-Agent Debate: Two recent works investigate and show that multi-agent debate is an effective way to encourage divergent thinking in LLMs [29] and to improve the factuality and reasoning of LLMs [16]. In both works, multiple LLM inference instances are constructed as multiple agents to solve problems with agent debate. Each agent is simply an LLM inference instance, while no tool or human is involved. The conversation also needs to follow a pre-defined order.

## major agents projects

- AutoGPT
	- https://twitter.com/karpathy/status/1642598890573819905
		- Next frontier of prompt engineering imo: "AutoGPTs" . 1 GPT call is just like 1 instruction on a computer. They can be strung together into programs. Use prompt to define I/O device and tool specs, define the cognitive loop, page data in and out of context window, .run().
		- Interesting non-obvious note on GPT psychology is that unlike people they are completely unaware of their own strengths and limitations. E.g. that they have finite context window. That they can just barely do mental math. That samples can get unlucky and go off the rails. Etc.
		- (so I'd expect the good prompts to explicitly address things like this)
		- tool list https://github.com/Significant-Gravitas/Auto-GPT/blob/ecf2ba12db11ff19bce359b842f810f0e2d09d6a/autogpt/prompt.py#L47
	- another variant https://replit.com/@zahid/AutoGPT-An-Autonomous-GPT-4-Experiment with [short demo video](https://twitter.com/LinusEkenstam/status/1646095934177124353)
- BabyAGI
	- smallest version https://replit.com/@YoheiNakajima/babyagi?v=1
	- write up https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/
	- https://github.com/yoheinakajima/babyagi
		- https://github.com/yoheinakajima/babyagi/blob/main/inspired_projects.md
	- UI https://github.com/miurla/babyagi-ui
- Microsoft JARVIS https://github.com/microsoft/JARVIS
	- seems an extension of the hugginggpt
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
	- langchain agent webinar https://www.crowdcast.io/c/46erbpbz609r
		- summary https://twitter.com/jh_damm/status/1646233627661828109
- Huggingface transformers agents thing
- Fixie AI
	- https://blog.fixie.ai/introducing-fixie-ai-a-new-way-to-build-with-large-language-models-d4e1aeee6b81?gi=33c317b7b536
	- [**Fixie.ai**](http://fixie.ai/), a new platform for building, hosting, and scaling Smart Agents that extend the power of Large Language Models with connections to your own data sources, systems, and tools. Think of Fixie like a way of building a ChatGPT-like experience that can be infinitely extended with new capabilities and interfaces to any software system.
- ai agents https://youtu.be/g0muj39vCCY
	- avalon - 3D reinforcement learning environment- simulation to test agent behavior
	- curiosity
	- long horizon planning
	- ability to learn very quickly
	- spending more time at inference is equal to 100,000 time at training. analogous to chain of thought just needing more tokens 30mins mark
- langchain https://twitter.com/hwchase17/status/1629929531678265344
	- problems with agents - hallucinating answer while showing steps https://vgel.me/posts/tools-not-needed/ if poorly implemented 
- minion ai https://twitter.com/alexgraveley/status/1631693523748519940
- multi-on
- [TaxyAI: Open-source browser automation with GPT-4](https://github.com/TaxyAI/browser-extension) ([github.com/taxyai](https://news.ycombinator.com/from?site=github.com/taxyai))
- embra ai
	- agent on desktop
- Uni ai - embra competitor
		- https://news.ycombinator.com/user?id=deet
- SLAPA approach
	- Recent work, such as Toolformer, have demonstrated that language models can teach themselves WHEN to use tools. With SLAPA, we show that language models can also teach themselves HOW to use tools!
	- learns how to use tools before using them by searching docs
	- When given a task, SLAPA knows to search for the API documentation and learn all the information. Then he create API calls. If they don't work, he learns from his mistake and tries again.
	- https://twitter.com/DYtweetshere/status/1631349179934203904
- Autogen: https://github.com/microsoft/autogen
	- https://arxiv.org/pdf/2308.08155v1.pdf
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


## low code agents

- https://magicloops.dev/
- https://flowiseai.com/ - visual langchain builder
- 

## OSS agent implementations

- Wove https://github.com/zckly/wove long running workflows 
- Smallville - similar to the Simulacra paper https://github.com/nickm980/smallville

## Robotics

https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/

there was a google thing as well. something like XP1?



## agent platforms

- https://superagent.sh/
- e2b

## Openai Function Calling

- read blogpost [https://openai.com/blog/function-calling-and-other-api-updates](https://openai.com/blog/function-calling-and-other-api-updates)
- function calling
    - [https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb)
    - [https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)
- [https://news.ycombinator.com/item?id=36480570](https://news.ycombinator.com/item?id=36480570)
- [https://twitter.com/jaredpalmer/status/1673350963191508993](https://twitter.com/jaredpalmer/status/1673350963191508993)
- [https://github.com/steven-tey/chathn](https://github.com/steven-tey/chathn)
    - [https://github.com/steven-tey/chathn/blob/8c62c12079d9e881444d03c67359d1ff4f8032c7/app/api/chat/functions.ts#L114](https://github.com/steven-tey/chathn/blob/8c62c12079d9e881444d03c67359d1ff4f8032c7/app/api/chat/functions.ts#L114)

## agent collaboration

- https://arxiv.org/abs/2307.05300v2 # Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration
- https://github.com/OpenBMB/ChatDev ChatDev stands as a virtual software company that operates through various intelligent agents holding different roles, including Chief Executive Officer , Chief Product Officer , Chief Technology Officer , programmer , reviewer , tester , art designer


## finetuning for agents/reasoning

- AgentTuning: Enabling Generalized Agent Abilities for LLMs https://arxiv.org/pdf/2310.12823.pdf
	- We construct AgentInstruct, a lightweight instruction-tuning dataset containing high-quality interaction trajectories. 
	- We employ a hybrid instruction-tuning strategy by combining AgentInstruct with open-source instructions from general domains. 
	- AgentTuning is used to instruction-tune the Llama 2 series, resulting in AgentLM.
	- Our evaluations show that AgentTuning enables LLMs' agent capabilities without compromising general abilities. The AgentLM-70B is comparable to GPT-3.5-turbo on unseen agent tasks, demonstrating generalized agent capabilities.