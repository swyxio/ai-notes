
## openai

- harvey case study https://x.com/gabepereyra/status/1775207692841488542?s=20
	- - 93% preferred vs. ChatGPT by BigLaw attorneys - 87% more accurate case citations
- more improvements to OpenAI's Fine-Tuning API & additional info on its Custom Models program [https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program "https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program")
	- Epoch-based Checkpoint Creation: Automatically produce one full fine-tuned model checkpoint during each training epoch, which reduces the need for subsequent retraining, especially in the cases of overfitting
	- Comparative Playground: A new side-by-side Playground UI for comparing model quality and performance, allowing human evaluation of the outputs of multiple models or fine-tune snapshots against a single prompt
	- Third-party Integration: Support for integrations with third-party platforms (starting with Weights and Biases this week) to let developers share detailed fine-tuning data to the rest of their stack
	- Comprehensive Validation Metrics: The ability to compute metrics like loss and accuracy over the entire validation dataset instead of a sampled batch, providing better insight on model quality
	- Hyperparameter Configuration: The ability to configure available hyperparameters from the Dashboard (rather than only through the API or SDK) 
	- Fine-Tuning Dashboard Improvements: Including the ability to configure hyperparameters, view more detailed training metrics, and rerun jobs from previous configurations

## frontier models

- Anthropic
	- [Claude - tool use now in beta](https://twitter.com/AnthropicAI/status/1775979802627084713)

## open models

- Cohere Command R+: [@cohere](https://twitter.com/cohere/status/1775878850699808928) released Command R+, a 104B parameter model with 128k context length, open weights for non-commercial use, and strong multilingual and RAG capabilities. It's available on the [Cohere playground](https://twitter.com/cohere/status/1775878883268509801) and [Hugging Face](https://twitter.com/osanseviero/status/1775882744792273209). [Aidan tweet](https://twitter.com/aidangomez/status/1775878606108979495)
	-   **Optimized for RAG workflows**: Command R+ is [optimized for RAG](https://twitter.com/aidangomez/status/1775878606108979495), with multi-hop capabilities to break down complex questions and strong tool use. It's integrated with [@LangChainAI](https://twitter.com/cohere/status/1775931339361149230) for building RAG applications.
	-   **Multilingual support**: Command R+ has [strong performance](https://twitter.com/seb_ruder/status/1775882934542533021) across 10 languages including English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Arabic, and Chinese. The SonnetTokenizer is [efficient for non-English text](https://twitter.com/JayAlammar/status/1775928159784915229).
- [Stable Audio 2.0](https://x.com/StabilityAI/status/1775501906321793266?s=20) - a new model capable of producing high-quality, full tracks with coherent musical structure up to three minutes long at 44.1 kHz stereo from a single prompt.


## memes

- suno memes
	- https://x.com/goodside/status/1775713487529922702