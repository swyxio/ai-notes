# prompt-eng

notes for prompt engineering

## Motivational Use Cases

- video
  - img2img of famous movie scenes ([lalaland](https://twitter.com/TomLikesRobots/status/1565678995986911236))
  - virtual fashion ([karenxcheng](https://twitter.com/karenxcheng/status/1564626773001719813))
  - evolution of scenes ([xander](https://twitter.com/xsteenbrugge/status/1558508866463219712))
  - outpainting https://twitter.com/orbamsterdam/status/1568200010747068417?s=21&t=rliacnWOIjJMiS37s8qCCw
  - image to video with rotation https://twitter.com/TomLikesRobots/status/1571096804539912192
- gpt3 applications
  - text to graphviz https://twitter.com/goodside/status/1561549768987496449?s=21&t=rliacnWOIjJMiS37s8qCCw
  - suspedning to python for math   
    - https://twitter.com/sharifshameem/status/1414029295043764226?lang=en
    - https://twitter.com/amasad/status/1568824744367259648

## Top Prompt Engineering Reads

- https://www.gwern.net/GPT-3#prompts-as-programming
- beginner
  - https://medium.com/nerd-for-tech/prompt-engineering-the-career-of-future-2fb93f90f117
  - https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/?utm_source=share&utm_medium=ios_app&utm_name=iossmf
- https://creator.nightcafe.studio/vqgan-clip-keyword-modifier-comparison VQGAN+CLIP Keyword Modifier Comparison
We compared 126 keyword modifiers with the same prompt and initial image. These are the results.
  - https://creator.nightcafe.studio/collection/8dMYgKm1eVXG7z9pV23W
- Google released PartiPrompts as a benchmark: https://parti.research.google/ "PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as part of this work. P2 can be used to measure model capabilities across various categories and challenge aspects."

## Tooling

- Deforum Diffusion https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb
- Disco Diffusion https://news.ycombinator.com/item?id=32660138
- [Edsynth](https://www.youtube.com/watch?v=eghGQtQhY38) and [DAIN](https://twitter.com/karenxcheng/status/1564635828436885504) for coherence
- [FILM: Frame Interpolation for Large Motion](https://film-net.github.io/) ([github](https://github.com/google-research/frame-interpolation))
- [Depth Mapping](https://github.com/compphoto/BoostingMonocularDepth)
  - examples: https://twitter.com/TomLikesRobots/status/1566152352117161990
- Art program plugins
  - Krita: https://github.com/nousr/koi
  - Photoshop: https://old.reddit.com/r/StableDiffusion/comments/wyduk1/show_rstablediffusion_integrating_sd_in_photoshop/
  - Figma: https://twitter.com/RemitNotPaucity/status/1562319004563173376?s=20&t=fPSI5JhLzkuZLFB7fntzoA
  - collage tool https://twitter.com/genekogan/status/1555184488606564353

Live updated list: https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/

## Communities

- StableDiffusion Discord https://discord.com/invite/stablediffusion
- Deforum Discord https://discord.gg/upmXXsrwZc
- Lexica Discord https://discord.com/invite/bMHBjJ9wRh
- Midjourney
- https://promptbase.com/
- Prompt Galleries
  - https://arthub.ai/
  - https://lexica.art/
  - https://dallery.gallery/

## Stable Diffusion

stable diffusion specific notes

Main: https://github.com/CompVis/stable-diffusion

Required reading:
- param intuitionhttps://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
- CLI commands https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/#script-options

### Distros

- Bundled Distros
  - https://www.charl-e.com/
  - https://github.com/divamgupta/diffusionbee-stable-diffusion-ui
- Web Distros
  - https://www.mage.space/
- Twitter Bots
  - https://twitter.com/diffusionbot
  - https://twitter.com/m1guelpf/status/1569487042345861121

### SD Major forks

https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/

Forks 

- https://github.com/basujindal/stable-diffusion 
  This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed.
- https://github.com/hlky/stable-diffusion ([here is](https://www.reddit.com/r/StableDiffusion/comments/x28a76/stable_diffusion_web_ui/) another fork that might be better)
  - adds a bunch of features - GUI/webui, [textual inversion](https://textual-inversion.github.io/), [upscalers](https://github.com/hlky/stable-diffusion-webui/wiki/Upscalers), mask and crop, img2img editor, word seeds, prompt weighting
    - doesn't work on Mac https://github.com/hlky/stable-diffusion/issues/173
  - How to Fine-tune Stable Diffusion using Textual Inversion https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-textual-inversion-b995d7ecc095
  - https://github.com/AbdBarho/stable-diffusion-webui-docker
    - Run Stable Diffusion on your machine with a nice UI without any hassle! This repository provides the WebUI as a docker image for easy setup and deployment. Please note that the WebUI is experimental and evolving quickly, so expect some bugs.
    - doesnt work on m1 mac yet https://github.com/AbdBarho/stable-diffusion-webui-docker/issues/31
- https://github.com/lstein/stable-diffusion and https://github.com/magnusviri/stable-diffusion
  - An interactive command-line interface that accepts the same prompt and switches as the Discord bot.
  - A basic Web interface that allows you to run a local web server for generating images in your browser.
  - A notebook for running the code on Google Colab.
  - Support for img2img in which you provide a seed image to guide the image creation. (inpainting & masking coming soon)
  - Upscaling and face fixing using the optional ESRGAN and GFPGAN packages.
  - Weighted subprompts for prompt tuning.
  - Textual inversion for customization of the prompt language and images.
- https://github.com/bfirsh/stable-diffusion
  - works on M1 Macs - [blog](https://replicate.com/blog/run-stable-diffusion-on-m1-mac), [tweet](https://twitter.com/levelsio/status/1565731907664478209)
  - can also look at `environment-mac.yaml` from https://github.com/fragmede/stable-diffusion/blob/mps_consistent_seed/environment-mac.yaml
- https://github.com/harubaru/waifu-diffusion
  - nicer GUI for img2img


SD Tooling

- SD's DreamStudio https://beta.dreamstudio.ai/dream
- Midjourney + SD: https://twitter.com/EMostaque/status/1561917541743841280
- [Nightcafe Studio](https://creator.nightcafe.studio/stable-diffusion-image-generator)
- misc
  - (super super raw dont try yet) https://github.com/breadthe/sd-buddy

## SD Model values


- How SD works
  - https://huggingface.co/blog/stable_diffusion
  - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
  - https://twitter.com/johnowhitaker/status/1565710033463156739
  - https://twitter.com/ai__pub/status/1561362542487695360
- [Exploring 12 Million of the 2.3 Billion Images Used to Train Stable Diffusion’s Image Generator](https://waxy.org/2022/08/exploring-12-million-of-the-images-used-to-train-stable-diffusions-image-generator/)
  - explore: https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/images
  - search: https://haveibeentrained.com/ ([tweet](https://twitter.com/matdryhurst/status/1570143343157575680))


## SD Results

### Img2Img

- A black and white photo of a young woman, studio lighting, realistic, Ilford HP5 400
  - https://twitter.com/TomLikesRobots/status/1566027217892671488


## Hardware requirements

- https://news.ycombinator.com/item?id=32642255#32646761
  - For something like this, you ideally would want a powerful GPU with 12-24gb VRAM. 
  - A $500 RTX 3070 with 8GB of VRAM can generate 512x512 images with 50 steps in 7 seconds.

## SD vs DallE vs MJ

DallE banned so SD https://twitter.com/almost_digital/status/1556216820788609025?s=20&t=GCU5prherJvKebRrv9urdw
