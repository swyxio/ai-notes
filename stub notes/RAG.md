
https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/

- [Building RAG-based LLM Applications for Production (Part 1)](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) full guide from Anyscale 
- LlamaIndex
	- [Slide deck](https://docs.google.com/presentation/d/1uzhz1aFWbyXSrWBzQ1FPQWtVjMgJqAYGoGoVzEnNmAg/edit#slide=id.p) ([addl resources](https://twitter.com/jerryjliu0/status/1700531889239437784))
	- [every AI engineer should learn how to build RAG from scratch](https://twitter.com/jerryjliu0/status/1702345670563332340) 🧑‍🍳 - no packaged retrievers/chains allowed. We’re excited to launch a new low-level tutorial series showing how to build the following: ✅ data ingestion ✅ retrieval With low-level components only (LLMs, prompts, embeddings, vector stores, no indexes). There’s more on the way too: 💡 response synthesis 💡advanced retrieval algorithms (routing, auto-retrieval, etc.) 💡agent loops (ReAct, OpenAI)
	- https://twitter.com/jerryjliu0/status/1733530504572592363?s=12&t=90xQ8sGy63D2OtiaoGJuww
	- ![https://pbs.twimg.com/media/GA68toAboAAWbYj?format=jpg&name=small](https://pbs.twimg.com/media/GA68toAboAAWbYj?format=jpg&name=small)

https://www.hebbia.ai/

## Vector DBs

- [ Choosing vector database: a side-by-side comparison](https://benchmark.vectorview.ai/vectordbs.html) ([HN](https://news.ycombinator.com/item?id=37764489))
- "Vector Database 101" series that covers vector search and vector indexes as well: [https://zilliz.com/learn/what-is-vector-database](https://zilliz.com/learn/what-is-vector-database)
	


## evaluation

- https://github.com/explodinggradients/ragas
	- 1.  **Faithfulness**: measures the information consistency of the generated answer against the given context. If any claims are made in the answer that cannot be deduced from context is penalized.
	- **Context Relevancy**: measures how relevant retrieved contexts are to the question. Ideally, the context should only contain information necessary to answer the question. The presence of redundant information in the context is penalized.
	3.  **Context Recall**: measures the recall of the retrieved context using annotated answer as ground truth. Annotated answer is taken as proxy for ground truth context.
	4.  **Answer Relevancy**: refers to the degree to which a response directly addresses and is appropriate for a given question or context. This does not take the factuality of the answer into consideration but rather penalizes the present of redundant information or incomplete answers given a question.
	5.  **Aspect Critiques**: Designed to judge the submission against defined aspects like harmlessness, correctness, etc. You can also define your own aspect and validate the submission against your desired aspect. The output of aspect critiques is always binary.

## visualization

- arize pheonix https://twitter.com/ArizePhoenix/status/1684013772997013504