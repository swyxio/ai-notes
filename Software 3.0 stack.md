## high quality guides and tutorial

full stack deep learning - llm bootcamp: https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/

## no code

- prototyping
	- nat.dev
	- https://play.vercel.ai/
- prompt engineering
	- https://github.com/ianarawjo/ChainForge An open-source visual programming environment for LLM experimentation and prompt evaluation.
		- alternatives
		- [https://github.com/logspace-ai/langflow](https://github.com/logspace-ai/langflow) a UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows.
		- [https://github.com/FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) - visual langchain buidler
	- [vellum.ai](https://techcrunch.com/2023/07/11/prompt-engineering-startup-vellum-ai/) has a visual flow editor thing.  tools for prompt engineering, semantic search, version control, quantitative testing, and performance monitoring.

## highest code level

- owning the endpoint
	- OpenLM - https://github.com/r2d4/openlm OpenAI-compatible Python client that can call any LLM
- SDK wrappers
	- https://github.com/minimaxir/simpleaichat
	- https://github.com/vercel-labs/ai
- prompt tooling
	- langchain
		- https://www.pinecone.io/learn/langchain/
	- llamaindex
	- deepset [haystack](https://haystack.deepset.ai/)
	- guardrails 
	- [scale spellbook](https://twitter.com/russelljkaplan/status/1590183663819718658)
- vector databases

## llmops

- portkey https://twitter.com/jumbld/status/1648684887988117508?s=46&t=90xQ8sGy63D2OtiaoGJuww
- helicone 
- Ozone - prompt unit testing https://twitter.com/at_sushi_/status/1667004844153131008
- https://log10.io/ - pivoting to llm quality monitoring
- eval
	- https://github.com/BerriAI/bettertest https://twitter.com/ishaan_jaff/status/1665105582804832258
	- https://github.com/AgentOps-AI/agentops
	- [Baserun.ai](https://www.ycombinator.com/launches/JFc-baserun-ai-ship-llm-features-with-confidence)
	- [https://hegel-ai.com](https://hegel-ai.com/), [https://www.vellum.ai/](https://www.vellum.ai/), [https://www.parea.ai](https://www.parea.ai/), [http://baserun.ai](http://baserun.ai/), [https://www.trychatter.ai](https://www.trychatter.ai/), [https://talc.ai](https://talc.ai/), [https://github.com/BerriAI/bettertest](https://github.com/BerriAI/bettertest), [https://langfuse.com](https://langfuse.com/)
	- https://github.com/mr-gpt/deepeval
	- [Hegel AI Prompttools](https://news.ycombinator.com/item?id=36958175)
		- [Show HN: OpenLLMetry – OpenTelemetry-based observability for LLMs](https://github.com/traceloop/openllmetry) ([github.com/traceloop](https://news.ycombinator.com/from?site=github.com/traceloop)) https://news.ycombinator.com/item?id=37843907
	- https://github.com/promptfoo/promptfoo
	- https://benchllm.com/
	- [https://www.getscorecard.ai](https://www.getscorecard.ai/)
	- [https://arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688)
	- [https://withmartian.com](https://withmartian.com/)
	- [https://aihero.studio/](https://aihero.studio/)
- evals 
	- scorecard
	- https://www.arthur.ai/blog/introducing-arthur-bench
- data quality
	- cleanlab.ai
	- deepchecks <- bigger
	- lilac ai
	- gallileo
	- 

## typing/json structure libraries 

- Microsoft TypeChat https://news.ycombinator.com/item?id=36803124
- jsonformer
- lmql

## lower code level

- hugginface transformers https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt
- lightning https://twitter.com/_willfalcon/status/1665826619200614401
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- Skypilot https://github.com/skypilot-org/skypilot a framework for running LLMs, AI, and batch jobs on any cloud, offering maximum cost savings, highest GPU availability, and managed execution.

SkyPilot **abstracts away cloud infra burdens**:

- Launch jobs & clusters on any cloud
- Easy scale-out: queue and run many jobs, automatically managed
- Easy access to object stores (S3, GCS, R2)


## vector databases

comparisons https://thedataquarry.com/posts/vector-db-1/

https://news.ycombinator.com/item?id=36943318
![https://miro.medium.com/v2/resize:fit:720/format:webp/1*fqiIXM6cHcOoEh_4WX0o1A.png](https://miro.medium.com/v2/resize:fit:720/format:webp/1*fqiIXM6cHcOoEh_4WX0o1A.png)

- chroma
- pinecone
- weaviate
- qdrant
- [marqo vector search](https://news.ycombinator.com/item?id=37147140)
- postgres
	- supabase vector
	- problems with it:   
https://twitter.com/nirantk/status/1674110063286571008?s=46

https://nextword.substack.com/p/vector-database-is-not-a-separate
-   [Cloudflare launches vectorize,](https://blog.cloudflare.com/vectorize-vector-database-open-beta/) announced on September 27th, 2023
-   [MongoDB Atlas Vector Search](https://www.mongodb.com/blog/post/introducing-atlas-vector-search-build-intelligent-applications-semantic-search-ai) launched on June 22nd, 2023
-   [Databricks](https://www.databricks.com/company/newsroom/press-releases/databricks-introduces-new-generative-ai-tools-investing-lakehouse) announced on June 28th, 2023
-   [Oracle integrated vector database](https://www.oracle.com/news/announcement/ocw-integrated-vector-database-augments-generative-ai-2023-09-19/) announced on September 19th, 2023
-   [IBM to announce vector database preview in Q4 2023](https://newsroom.ibm.com/2023-09-07-IBM-Advances-watsonx-AI-and-Data-Platform-with-Tech-Preview-for-watsonx-governance-and-Planned-Release-of-New-Models-and-Generative-AI-in-watsonx-data)
-   of course, companies such as Elastic and Microsoft already had vector DB offerings much earlier.

ETL
- psychic.dev


fully vertically integrated RAG cloud
- vectara -29m raised and from former cloudera founder
- https://pezzo.ai - "enables you to build, test, monitor and instantly ship AI all in one platform, while constantly optimizing for cost and performance." - used by Meltwater CTO - from shack15
- https://www.pulze.ai maybe?

## infra

- https://mlfoundry.com/
- together.ai
- model hosting and finetuning
	- LLM Engine ([https://llm-engine.scale.com](https://llm-engine.scale.com/)) at Scale, which is our open source, self-hostable framework for open source LLM inference and fine-tuning. ([source](https://news.ycombinator.com/item?id=37492776))
	- replicate

## coding tools

- https://github.com/danielgross/localpilot
- https://github.com/continuedev/continue
- https://github.com/mudler/LocalAI
- https://vxtwitter.com/ex3ndr/status/1726863029919482167


## misc

- AI relational database https://github.com/georgia-tech-db/eva
- finetune industry
	- https://predibase.com/
- AI devtools
	- codegen.ai