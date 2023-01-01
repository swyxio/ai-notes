
prompt engineering techniques

## Code related prompts

- generate csv, test code, and readme https://twitter.com/goodside/status/1563989550808154113
- Rewrite regex + examples + unit tests https://twitter.com/goodside/status/1562233738863452160
- convert object to schemas, type assertions, and table conversion parsing https://twitter.com/goodside/status/1513265657944678401?s=20

## Programmatic

## k-shot prompts with JSON encoding

- "split sentinel" approach https://twitter.com/goodside/status/1564437905497628674?s=20

### You can't do math

https://twitter.com/goodside/status/1568448128495534081/photo/1

impls
- https://beta.openai.com/playground/p/WV3rBB5qKFR2O5MBDqYVBpL3?model=text-davinci-002
- https://replit.com/@amasad/gptpy?v=1
- sharif shameem direct version https://gist.github.com/Samin100/6cec8c3f9e5d68e0776fcac6e5ba86aa

```
You are GPT-3, and you can't do math.

You can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong,
answers.

So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we'll take care of the rest:

Question: ${Question with hard calculation.}
\```python
${Code that prints what you need to know}
\```

\```output
${Output of your code}
\```

Answer: ${Answer}

Otherwise, use this simpler format:

Question: ${Question without hard calculation} 
Answer: ${Answer}

Begin.
```


### get Google SERP results

https://twitter.com/goodside/status/1568532025438621697?s=20
https://cut-hardhat-23a.notion.site/code-for-webGPT-44485e5c97bd403ba4e1c2d5197af71d

### GPT3 doing Instruction templating of python GPT3 calls

- initial idea https://twitter.com/goodside/status/1609465914453377024?s=20
	- works in chatgpt https://twitter.com/goodside/status/1609489758584963073

```
Use this format:

\```
<python 3 shebang>
<module docstring>
<imports>
<dunders: by Riley Goodside; 2022 by author; MIT license>
<do not include email dunder>

<initialize dotenv>
<set key using OPENAI_API_KEY env var>

def complete(prompt: str, **openai_kwargs) -> str:
	<one-line docstring; no params>
	<use default kwargs: model=text-davinci-003, top_p=0.7, max_tokens=512> <note: `engine` parameter is deprecated>
	<get completion>
	<strip whitespace before returning>
\```

<as script, demo using prompt "English: Hello\nFrench:">
```

- grimes application https://twitter.com/goodside/status/1609363458037895169/photo/1

```
Use this format:
\```
<imports>
<initialize dotenv>
<read key from env "OPENAI_API_KEY">

def complete(prompt: str, **openai_kwargs) -> str:
	<one-line docstring>
	#`engine parameter is deprecated
	default_kwargs = {"model": "text-davinci-003", "max_tokens": 256, "top_p":0.7}
	openai_kwargs=default_kwargs | openai_kwargs
	<...>

def ask_chain_of_thought(question: str) -> str:
	<one-line docstring>
	cot_prompt_format = "Q: {question}\nA: Let's think step by step."
	extract_prompt_format = "{cot_prompt}{cot_completion} Therefore, the final answer (one letter in double-quotes) is:"
	<...>

def ask_consensus_cot(question:str, n=5) -> str:
	<one-line docstring>
	<call ask_chain_of_thought n times and return modal answer>

question = "What is the final character of the MD5 hash of the last digit of the release year of the Grimes album 'Visions'?" 
<print consensus answer>
```


### `guess()` function

https://twitter.com/goodside/status/1609436504702717952

remove the `\```` escapes:

```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
source = inspect.getsource (inspect.current frame())

def guess (what: str) -> str:
prompt = f"""\
Code:
\```
{source}
\```

Based on context, we could replace `guess({what!r})` with the string:
\```
"""

	return openai. Completion.create(
		prompt=prompt,
		stop="\\\"
		max_tokens=512,
		model="text-davinci-003",
		temperature=0,
	) ["choices"] [0] ["text"].strip()

# Test the guess function:
print(f"Apples are typically {guess('color')}.")
print (f"The drummer for The Beatles was {guess ('name')}.")
print("Pi is approximately {guess('pi')}, whereas e is approximately {guess('e')}.")
print (f"A paragraph-length explanation of the bubble sort would be: {guess('explanation')}")
```


## maybes

- gpt3 plays ipython https://twitter.com/goodside/status/1581337959856422912?s=20
- produce graphviz digraph https://twitter.com/goodside/status/1546335018225745920?s=20