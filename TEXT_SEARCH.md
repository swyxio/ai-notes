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
- https://www.hebbia.ai/
	- https://hebbia.medium.com/hebbia-raises-30-million-led-by-index-ventures-to-launch-the-future-of-search-e80038c05852
- Huberman search
	- https://twitter.com/rileytomasek/status/1603854647575384067?s=20
		- How does it work? Transcriptions are generated using Whisper and then embedded using the text-embedding-ada-002 model. The vectors are then stored in a pinecone vector database. A user's query is embedded and then used to find similar vectors in the database.
		- The "Ask" answer uses text-davinci-003 to answer the question given the search results, with instructions not to make stuff up.
		- https://github.com/rileytomasek/openai-fetch

## Tech notes

- How to build Semantic search distributed systems using python, pyspark, faiss and clip! A walk through on building a laion5B semantic search system.
	- https://rom1504.medium.com/semantic-search-at-billions-scale-95f21695689a
- https://www.deepmind.com/blog/gophercite-teaching-language-models-to-support-answers-with-verified-quotes