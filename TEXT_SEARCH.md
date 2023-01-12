<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [Search Products](#search-products)
- [Tech notes](#tech-notes)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Search Products

- Metaphor https://metaphor.systems/
- https://perplexity.ai Ask
	- new search interface that uses OpenAI GPT 3.5 and Microsoft Bing to directly answer any question you ask.
	- uses Codex https://news.ycombinator.com/item?id=34006542
	- examples https://twitter.com/perplexity_ai/status/1600551956551852032?s=20
- https://www.hebbia.ai/
	- https://hebbia.medium.com/hebbia-raises-30-million-led-by-index-ventures-to-launch-the-future-of-search-e80038c05852
- Huberman search
- Neeva AI https://neeva.com/blog/introducing-neevaai
- You AI
- seekai
	- Seek falls into the category of enterprise search engines known as “cognitive search.” Rivals include [Amazon Kendra](https://techcrunch.com/2020/05/11/amazon-releases-kendra-to-solve-enterprise-search-with-ai-and-machine-learning/) and Microsoft SharePoint Syntex, which draw on knowledge bases to cobble together answers to company-specific questions. Startups like [Hebbia](https://techcrunch.com/2022/09/07/hebbia-raises-30m-to-launch-an-ai-powered-document-search-tool/), Kagi, [Andi](https://techcrunch.com/2022/09/13/y-combinator-backed-andi-taps-ai-to-built-a-better-search-engine/) and [You.com](https://techcrunch.com/2022/07/14/you-com-raises-25m-to-fuel-its-ai-powered-search-engine/) also leverage AI models to return specific content in response to queries as opposed to straightforward lists of results.
- productized [https://addcontext.xyz](https://addcontext.xyz/)
	- https://twitter.com/rileytomasek/status/1603854647575384067?s=20
		- How does it work? Transcriptions are generated using Whisper and then embedded using the text-embedding-ada-002 model. The vectors are then stored in a pinecone vector database. A user's query is embedded and then used to find similar vectors in the database.
		- The "Ask" answer uses text-davinci-003 to answer the question given the search results, with instructions not to make stuff up.
		- https://github.com/rileytomasek/openai-fetch

## Tech notes

- How to build Semantic search distributed systems using python, pyspark, faiss and clip! A walk through on building a laion5B semantic search system.
	- https://rom1504.medium.com/semantic-search-at-billions-scale-95f21695689a
- https://www.deepmind.com/blog/gophercite-teaching-language-models-to-support-answers-with-verified-quotes
- https://haystack.deepset.ai/overview/intro 
- 3) [Transformer Memory as a Differentiable Search Index (“DSI”)](https://arxiv.org/abs/2202.06991)
- openai embeddings and pinecone 
	- arxiv search https://twitter.com/tomtumiel/status/1611729847700570118?s=20&t=esNCMGOrghGYObzQee1Hzg
- weaviate vecot rsearch https://twitter.com/CShorten30/status/1612081726041518080?s=20