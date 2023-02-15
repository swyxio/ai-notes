# AI Hackathon Stack

> ⚠️ This is a **very** new/immature list, created for [the Latent Space Demo Day](https://lspace.swyx.io/p/demo-day-2023). The category labels not that well thought through. Please [get in touch](https://discord.gg/gR6yP6wbfq) if you have better ideas for how to organize this, it is welcome.

Below is a curated list of useful tools and examples for developers tackling AI Hackathons, together with useful hackathon-centric offers and templates for you to get started quickly.

If you represent a tool or vendor, please feel free to send in a PR for your tool, but note that we may reword or reject your submission based on subjective fit.



## AI Infra and Tooling

This category is for **infra and tools** catering to AI app developers, in contrast to **apps** (below) which have other kinds of end users in mind.

- **Tools to explore**
  - **Serverless GPUs**
    - https://exafunction.com/ Exafunction optimizes your deep learning inference workload, delivering up to a 10x improvement in resource utilization and cost.
    - https://www.banana.dev/ Scale your machine learning inference and training on serverless GPUs.
    - https://brev.dev/ The simplest way to create a dev environment with a GPU. Don't worry about dependencies, CUDA, SSH, or anything else. Up to 94% cheaper GPUs than AWS.
    - https://lambdalabs.com/ GPU cloud built for deep learning
Instant access to the best prices for cloud GPUs on the market. No commitments or negotiations required. Save over 73% vs AWS, Azure, and GCP. Configured for deep learning with PyTorch®, TensorFlow, Jupyter
    - _Seeking: hackathon-relevant examples and tutorials for each of these examples_
  - **Model Serving**
    - https://www.baseten.co/ serverless backend for building ML-powered applications. Build apps with auto-scaling, GPU access, CRON jobs, and serverless functions.
    - https://replicate.com/ Run models in the cloud at scale.
    - https://modal.com run or deploy machine learning models, massively parallel compute jobs, task queues, web apps, and much more, without your own infrastructure.. Example [serving Stable Diffusion API](https://modal.com/docs/guide/ex/stable_diffusion_slackbot)
    - _Seeking: hackathon-relevant examples and tutorials for each of these examples_
  - **Embeddings**
    - [**Chroma**](https://www.trychroma.com/): Chroma is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs. [Docs](https://docs.trychroma.com/), [Discord](https://discord.gg/MMeYNTmh3x), [@ the founders](https://twitter.com/atroyn/status/1625568377766035456?s=20&t=m96ilnMSQjoyjVmp_kQHZA)
    - [**Pinecone**](https://www.pinecone.io/): The Pinecone vector database makes it easy to build high-performance vector search applications. [Docs](https://www.pinecone.io/docs/), [Events/Forum/Showcase](https://www.pinecone.io/community/).
    - _Seeking: comparison articles between these options_
  - _have something to add? send a PR!_
- **Hackathon Entry Examples**
  -  a key-value store to enable long-term memory in language model conversations ([tweet](https://twitter.com/russelljkaplan/status/1616955361705197568?s=20&t=KIszRKntkT4Y-I-WwKI8Mg))

## LLM/Prompt Engineering Apps

- **Tools to explore**
  - **OpenAI** needs no introduction. [Cookbook](https://github.com/openai/openai-cookbook/), [Docs](https://platform.openai.com/docs/introduction/overview)
  - [**LangChain**](https://github.com/hwchase17/langchain/) - Building applications with LLMs through composability. [Discord](https://discord.gg/6adMQxSpJS)
  - [**Lambdaprompt**](https://github.com/approximatelabs/lambdaprompt) - Build, compose and call templated LLM prompts
  - _have something to add? send a PR!_
- **Hackathon Entry Examples**
  - Automatic permit application generation for climate tech companies & carbon dioxide removal ([tweet](https://twitter.com/russelljkaplan/status/1616957750940176384?s=20&t=frXEVPqaJUjMPJOhbD9AUg))
  - a personalized learning curriculum generator ([tweet](https://twitter.com/russelljkaplan/status/1616955367728222208?s=20&t=KIszRKntkT4Y-I-WwKI8Mg))

## Code AI tools

- **Tools to explore**
  - 
  - _have something to add? send a PR!_
- **Hackathon Entry Examples**
  - 🏆 [GPT is all you need for backend](https://github.com/TheAppleTucker/backend-GPT): a backend and database that is entirely LLM-powered. ([tweet](https://twitter.com/karpathy/status/1618311660539904002))
  - GPT-3 Auditor: scanning code for vulnerabilities with LLMs. https://github.com/handrew/gpt3-auditor

## Audio/Visual/Multimodal Apps

- **Tools to explore**
  - https://www.synthesia.io/ AI video and voice creation
  - _have something to add? send a PR!_
- **Hackathon Entry Examples**
  - HouseGPT generates raw MIDI data directly from few-shot prompted GPT-3 to create 🎶 house music 🎶 🔊 ([tweet](https://twitter.com/russelljkaplan/status/1616997544307089408?s=20&t=frXEVPqaJUjMPJOhbD9AUg))
  - [Rap Battle](https://twitter.com/russelljkaplan/status/1617070021406265345?s=20&t=frXEVPqaJUjMPJOhbD9AUg) - Pick any two people and it will generate a rap battle on the fly, using GPT-3 for lyrics, wavenet for vocals, and stable diffusion for the avatars. 
  - Game of Life, where each alive cell is a whimsical happy Stable Diffusion image and each dead cell is an eerie, dark Stable Diffusion image, all of which evolve over time. ([tweet](https://twitter.com/russelljkaplan/status/1616955356189687810?s=20&t=KIszRKntkT4Y-I-WwKI8Mg))
  - [Gptcommit: Never write a commit message again (with the help of GPT-3)](https://zura.wiki/post/never-write-a-commit-message-again-with-the-help-of-gpt-3/)
  - [santacoder typosaurus]([url](https://twitter.com/corbtt/status/1616270918774575105)) - semantic linter that spots errors in code
  - stackoverflow.gg https://twitter.com/bentossell/status/1622513022781587456
  - Buildt -  AI-powered search allows you to find code by searching for what it does, not just what it is. https://twitter.com/AlistairPullen/status/1611011712345317378

## Misc Useful Resources

Looking for tutorials, writeups, guides, inspo, requests for startups.

- Hackathon Recaps for Inspiration
  - Scale AI Hackathon https://scale.com/blog/generative-ai-hackathon
  - AssemblyAI Hackathon https://twitter.com/AssemblyAI/status/1602717569659682816
- _have something to add? send a PR!_
