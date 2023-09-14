## Timeline

- March 23 - plugins preview https://openai.com/blog/chatgpt-plugins
	- gdb video clip https://twitter.com/gdb/status/1638971232443076609
	- It's writing & executing MoviePy code ([https://zulko.github.io/moviepy/](https://t.co/SHPNZOLCHp)), which uses ffmpeg under the hood. So the time here is basically all waiting for ffmpeg to run (a big step up from the usual bottleneck of looking up the set of arguments to do what I want!).
- Jul 6-8 - GA release https://help.openai.com/en/articles/6825453-chatgpt-release-notes
- Jul 10 - Latent Space Writeup: https://www.latent.space/p/code-interpreter#details

## Code Interpreter system prompt

https://chat.openai.com/share/38301a21-bf09-4274-848b-b9cd5523d461

and requirements.txt https://twitter.com/swyx/status/1677589535587467264
([annotated](https://chat.openai.com/share/a4336af0-6faa-4ffb-b5b0-a3bd7ef8f3aa))


## Confirmed Capabilities at Launch

- Video manipulation
	- using moviePy https://twitter.com/gdb/status/1638971232443076609
- Accepted filetypes
	- .zip files: it can unzip files and continue to open up and inspect those files
		- "I apologize for the oversight. The variable `unzip_folder` was also lost in the reset. We'll need to extract the data from the zip file again before proceeding"
	- .csv files
	- .xls files https://twitter.com/emollick/status/1652170706312896512
	- .png files
		- lightweight image editing and resizing
	- .mov files
- Can draw https://twitter.com/swyx/status/1677567255377494016?s=20
	- but has challenges
	- create a graphviz chart - of code https://chat.openai.com/share/adc73608-4c80-497b-ad03-31de28bfa919
- Can map using folium
	- https://twitter.com/emollick/status/1677570911124242433?s=20
	- render the folium map to an image and show the image
- Data analyst
	- **Open ended analysis**: 
		- "Here is some data on superhero powers, look through it and tell me what you find"
		- "I am interested in doing some predictive modelling, where we can predict what powers a hero might have based on other factors."
	- **Joins**: "Could you first combine the two datasets"
	- **Integrity checks**: "... and confirm that they combined properly"
	- **Data Cleaning**: "does the data need cleaning in any way?"
	- **Data Preprocessing**: "Great! Cluster analysis can help us group similar superheroes together based on their attributes. Before we start, we need to preprocess our data. This will involve the following steps:" https://chat.openai.com/share/770ab170-8fed-402f-850f-e6d7273e77cd
	- **Summary of analysis**: "what is your summary of all this analysis, in bullet points?"
	- **Create Interactive Dashboard** https://emollick.github.io/Superhero/
		- and graphs https://emollick.github.io/3Dmusic/
		- and maps https://github.com/emollick/GPTflightmap.github.io
	- Dataviz
		- ![https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55543b86-b8ec-45db-84ec-649fe0237097_3208x2000.png](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55543b86-b8ec-45db-84ec-649fe0237097_3208x2000.png)
- Can upload more libraries and dependencies
	- how to: https://github.com/SkalskiP/awesome-chatgpt-code-interpreter-experiments#install-external-python-packages
	- running GPT-2 with GGML: https://twitter.com/sdand/status/1678476411416498178
- Behavior
	- **Giving options, and pausing for confirmation** 
		- Great! Before we proceed, we need to further refine the scope of our predictive task. As you might notice, we have a large number of potential powers (labels) to predict, which could make the problem very complex. Would you like to predict all possible powers, or focus on a specific subset of powers?
		- Another important point to note is that is our current dataset is quite large and might be too complex for certain algorithms. We might need to simplify our problem further by reducing the number of features. This could be done through a dimensionality reduction technique or simply selecting a subset of the most relevant features.
		- Finally, we need to choose a suitable model for our multi-label classification task. Some possibilities include decision trees, random forests, or neural networks. Each of these options has its own strengths and weaknesses, and the best choice often depends on the specific characteristics of the dataset. Please let me know how you would like to proceed.
		- "pick which you think is best, be clear about strengths and weaknesses"
		- Outliers can have a large influence on machine learning models, particularly those that use distance measures like Random Forest. Given this, we could:
			-   Remove these outliers from the data.
			-   Winsorize the data, which means capping the extreme values at a certain threshold.
			-   Leave the data as-is if we believe the extreme values are accurate and important to the model.
		- Which approach would you like to take?
	- Recovery from minor data issues
		- `It seems like there are still some values in the data that are causing issues. In the error message, it's mentioned that the input contains NaN, infinity, or a value too large for dtype('float32').`
		- `It appears there are no missing (NaN) or infinite values in the dataset after our preprocessing steps. The error during model training might be due to a different issue.`
		- `Looking at the unique values in the 'Height' and 'Weight' columns, we see some extreme values. For example, heights of 975 cm and 876 cm, and weights of 900 kg and 855 kg are unusually high for humans and most known superhero characters. These could be misrecorded data, placeholder values for unknowns, or correct data for exceptionally large characters.`
		- `IndexError: index 0 is out of bounds for axis 0 with size 0`-  `I apologize for the confusion. It seems like there was a misunderstanding in my previous step. I attempted to use the encoded name (a numerical value) to fetch the original name from the preprocessed data, but this approach was not correct because the numerical value does not represent an index in the original data.`
	- Requesting for more data
		- https://twitter.com/emollick/status/1654729461458190338/photo/2
- Environment
	- Preemption recovery? "I'm sorry, but it seems there was a disruption in the code execution environment causing a reset."
		- "Given the repeated resets in the code execution state, I think the most efficient way forward is to write a self-contained script that does not rely on any variables from earlier in the session."
	- Dynamically imports modules
	- **Catastrophic recovery and retry**: "Let's start from the beginning by extracting the data from the zip file again. I'll make sure to correctly specify the locations this time." https://chat.openai.com/share/770ab170-8fed-402f-850f-e6d7273e77cd
	- can you display this .png inline?
		- from IPython.display import Image # Display the generated flowchart Image(filename='/mnt/data/flowchart_enhanced.png')
		- <IPython.core.display.Image object>
	- KeyboardInterrupt: after long execution https://chat.openai.com/share/adc73608-4c80-497b-ad03-31de28bfa919
	- Image processing with PIL: "make the background of this image black"
	- Cv2 library for computer vision, pytesseract for OCR
	

## Deprecated Research Preview capabilities
- running subprocess.run (according to Simonw)



## Things that Code Interpreter doesn't do well

- display of tables is purely LLM generated and slow. need to swap to component and dump code output thru the component
- "the code execution state was reset" messages quite often https://chat.openai.com/share/adc73608-4c80-497b-ad03-31de28bfa919
- environment recovery is bad: https://twitter.com/swyx/status/1677571294416756737?s=20
- it frequently hallucinates dependencies which are super common
	- `NameError: name 'df' is not defined`
	- `NameError: name 'pd' is not defined`
	- `NameError: name 'zipfile' is not defined`
	- `NameError: name 'np' is not defined`
	- `NameError: name 'combinations' is not defined` I apologize for the oversight. It seems I forgot to import the `combinations` function from the `itertools` module`
	- `NameError: name 'ImageDraw' is not defined`. I apologize for the error, it appears I forgot to import the `ImageDraw` module from the `PIL` package. L
- no internet access
	- To proceed, we would need the descriptions of each package. These can typically be obtained from the package's documentation or from PyPI (the Python Package Index). However, fetching data from the internet is not permitted in the current environment.
- it doesnt "fill in the blanks" from LLM knowledge - instruction to "guess the data you don't have" fails
- thinks it doesnt have folium... but it does
- render the folium map to an image and show the image
	- Please note that due to the limitations of my current environment, I'm unable to convert the HTML file to an image directly within this notebook. However, you can do this in a local Python environment using tools like `imgkit` or `selenium`.
- Unfortunately, without more data and given the constraints of this environment, it's not possible to perform a detailed spatial analysis of the crime data. For a complete analysis, I would recommend using a GIS (Geographic Information System) software or library (like `geopandas` in Python) and obtaining more detailed spatial data.
- clustering and naming
	- cant call openai api to embed text, doesnt have transformers, and no UMAP

## collections of code interpreter capabilities and other lists

- manual Chat GPT Code Interpret git scraper. Includes libraries, env vars, complete filepath listing. Can track changes. Todo: system package list, home folder content etc,. added some file system extracts like the `ace_client` which is part of the jupyter notebook setup https://github.com/tomviner/scrape-ai/tree/main#chatgpt-code-interpreter
- https://github.com/SkalskiP/awesome-chatgpt-code-interpreter-experiments

## open source clones

- https://github.com/approximatelabs/datadm
- shroominic https://twitter.com/hwchase17/status/1680620252433117184?s=20
	- https://blog.langchain.dev/code-interpreter-api/
	- https://github.com/shroominic/codeinterpreter-api
	- https://twitter.com/arghzero/status/1691768136431988783
- https://github.com/e2b-dev/llm-code-interpreter
- https://github.com/tomviner/scrape-ai/tree/main/openai/code-interpreter/filesystem
- openinterpreter