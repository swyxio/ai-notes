## GPT specific notes

My best timeline of GPT efforts is listed here: https://lspace.swyx.io/p/open-source-ai

GPT3 applications:
  - text to graphviz https://twitter.com/goodside/status/1561549768987496449?s=21&t=rliacnWOIjJMiS37s8qCCw
  - suspending to python for math   
    - https://twitter.com/sharifshameem/status/1414029295043764226?lang=en
    - https://twitter.com/amasad/status/1568824744367259648
    - and API's https://twitter.com/sergeykarayev/status/1569377881440276481
  - Amelia paragraph sumarizer https://twitter.com/Wattenberger/status/1412480516268437512
  - Karina Nguyen Synevyr https://twitter.com/karinanguyen_/status/1566884536054677506
  - Lex.page
  - https://github.com/louis030195/obsidian-ava obsidian integration
  - https://humanloop.com/ Playground that brings variable interpolation to prompts and lets you turn them into API endpoints. Once you're deployed, it also lets you collect past generations along with user behavioral feedback for fine-tunes.
  - https://www.everyprompt.com/ extends Playground in a similar way: Putting variables in prompts and giving you a single button to go from prompt to API. Has nice developer-oriented touches in the UI too — e.g. displaying invisible chars as ghosts.
  - explaining code diffs https://app.whatthediff.ai/dashboard
  - LangChain https://twitter.com/hwchase17/status/1588183981312180225
    - implements and lets you easily compose many published LLM prompting techniques. Implements self-asking, web search, REPL math, and several of my own prompts.
    - All relevant chains now have a "verbose" option to highlight text according to the model or component (SQL DB, search engine, python REPL, etc) that it's from. 
  - https://dust.tt/ gives a collapsible tree UI for representing k-shot example datasets, prompt templates, and prompt chaining with intermediate JS code. Replaces a lot of code around prompt APIs. 
  - playing chess??? https://twitter.com/Raza_Habib496/status/1591514638520311809
  - chrome extension
    - https://github.com/giosilvi/GPT-Prompter https://www.reddit.com/r/GPT3/comments/wa2db1/new_chrome_extension_for_gpt3_for_fast_and_custom/
  - simulating people
    - https://jack-clark.net/2022/10/11/import-ai-305-gpt3-can-simulate-real-people-ai-discovers-better-matrix-multiplication-microsoft-worries-about-next-gen-deepfakes/
  - Making stories with characters https://medium.com/@turc.raluca/introducing-rick-and-mortify-a14e56a8cb67


## Top GPT3 Prompt Engineering Reads

- Overview
  - https://www.gwern.net/GPT-3#prompts-as-programming
  - https://andymatuschak.org/prompts/
- Beginner
  - go through all the GPT3 examples https://beta.openai.com/examples
- Intermediate
  - and deploy GPT2 https://huggingface.co/gpt2
  - play with the smaller GPT3 models https://beta.openai.com/docs/models/gpt-3
  - technique: self-asking, two step prompts https://twitter.com/OfirPress/status/1577302998136819713
    - chain of thought prompting https://twitter.com/OfirPress/status/1577303423602790401
  - Prompt structure with extensive examples
    - review feedback extraction https://www.youtube.com/watch?v=3EjtHs_lXnk&t=1009s
  - Ask Me Anything prompting
    - https://twitter.com/_akhaliq/status/1577838951905247235
  - using gpt3 for text2image prompts https://twitter.com/fabianstelzer/status/1554229347506176001
- Advanced
  - write a blogpost with GPT3 https://www.youtube.com/watch?v=NC7990PmDfM
  - integrating Google Search with GPT3: https://twitter.com/OfirPress/status/1577302733383925762
  - teach AI how to fish - You are X, you can do Y: https://github.com/nat/natbot/blob/main/natbot.py
  - play with gpt-neoX and gpt-j https://neox.labml.ai/playground
  - defense against prompt injection https://twitter.com/goodside/status/1578278974526222336
  - whatever the f this is https://twitter.com/goodside/status/1578614244290924545
- Academics
  - P3: Public pool of prompts https://huggingface.co/datasets/bigscience/P3
    - and Promptsource https://github.com/bigscience-workshop/promptsource
  - Real-world Annotated Few-shot Tasks (RAFT) dataset https://huggingface.co/datasets/ought/raft
  - Study chain of thought reasoning https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html
    -  and UL2 20B https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html
  - building GPT-JT: https://www.together.xyz/blog/releasing-v1-of-gpt-jt-powered-by-open-source-ai

## How GPT works

- https://github.com/karpathy/minGPT
  - announcement https://twitter.com/karpathy/status/1295410274095095810  
  - used in https://www.mosaicml.com/blog/gpt-3-quality-for-500k

## Don't call it generative

- Reasoning: https://twitter.com/alexandr_wang/status/1588933553290870785
- Understanding: https://twitter.com/EMostaque/status/1585903983180537856

## GPT3 versions

- GPT3 advanced a lot through 2020-2022 https://twitter.com/tszzl/status/1572350675014516738
- Eleuther's GPT-J-6B, GPT-NeoX
- GPT-JT for classification
  - https://news.ycombinator.com/item?id=33796158
  - https://huggingface.co/spaces/togethercomputer/GPT-JT
- GPT 3.5 (https://beta.openai.com/docs/model-index-for-researchers)
  - code-davinci-002 is a base model, so good for pure code-completion tasks
  - text-davinci-002 is an InstructGPT model based on code-davinci-002
  - text-davinci-003 is an improvement on text-davinci-002
  - InstructGPT https://openai.com/blog/instruction-following/
  - ChatGPT: https://openai.com/blog/chatgpt/
    - We’ve trained a model called ChatGPT which interacts in a conversational way. The dialogue format makes it possible for ChatGPT to answer followup questions, admit its mistakes, challenge incorrect premises, and reject inappropriate requests.  

## Specialized langauge models

- Scientific language models like Meta's Galactica exist. Commentary https://news.ycombinator.com/item?id=33614608

## Ethical issues

- Galactica fallout
  - https://twitter.com/Michael_J_Black/status/1593133722316189696
