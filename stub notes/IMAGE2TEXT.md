Reverse Prompt Engineering for images.

- [img2prompt](https://replicate.com/methexis-inc/img2prompt) Replicate by [methexis-inc](https://replicate.com/methexis-inc): Optimized for SD (clip ViT-L/14).
- https://www.img2prompt.io/
	- https://news.ycombinator.com/item?id=34705706 not very good?
- [CLIP Interrogator](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) by [@pharmapsychotic](https://twitter.com/pharmapsychotic): select ViTL14 CLIP model.
  - https://huggingface.co/spaces/pharma/sd-prism Sends an image in to CLIP Interrogator to generate a text prompt which is then run through Stable Diffusion to generate new forms of the original!

CLIP retrieval
- https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn5.laion.ai&index=laion5B&useMclip=false
- https://haveibeentrained.com/
- [LAION Clip retrieval](https://knn5.laion.ai/): search [laion-5b](https://laion.ai/blog/laion-5b/) dataset.
- [Datasette](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls)

SalesforceBLIP
- https://github.com/salesforce/BLIP
- https://github.com/Nutlope/alt-text-generator



GIT: A Generative Image-to-text Transformer for Vision and Language 
- https://twitter.com/natolambert/status/1612119135701303302?s=46&t=NcVvYe4-zKXX2KBHOi_C4w
- 

Other langages
- Chinese CLIP https://twitter.com/JustinLin610/status/1600524228201095169

Lip reading!
- https://huggingface.co/spaces/vumichien/lip_movement_reading


pulsr io and more from this thread
https://twitter.com/tunguz/status/1616190582606467089?s=46&t=eCig8-Pc5CuJQeXulVU7qQ


Flamingo model https://arxiv.org/abs/2204.14198


## VQA

LlaVA
- Visual Instruction Tuning [Haotian Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+H), [Chunyuan Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+C), [Qingyang Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+Q), [Yong Jae Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+Y+J)

> Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.

- https://llava-vl.github.io/