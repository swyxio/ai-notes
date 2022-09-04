# prompt-eng

notes for prompt engineering

## Motivational Use Cases

- video
  - img2img of famous movie scenes ([lalaland](https://twitter.com/TomLikesRobots/status/1565678995986911236))
  - virtual fashion ([karenxcheng](https://twitter.com/karenxcheng/status/1564626773001719813))

## Top Prompt Engineering Reads

- https://www.gwern.net/GPT-3#prompts-as-programming
- https://medium.com/nerd-for-tech/prompt-engineering-the-career-of-future-2fb93f90f117
- 
- https://creator.nightcafe.studio/vqgan-clip-keyword-modifier-comparison VQGAN+CLIP Keyword Modifier Comparison
We compared 126 keyword modifiers with the same prompt and initial image. These are the results.
  - https://creator.nightcafe.studio/collection/8dMYgKm1eVXG7z9pV23W

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


## Communities

- Deforum Discord https://discord.gg/upmXXsrwZc
- Midjourney
- https://promptbase.com/

## Stable Diffusion

stable diffusion specific notes

### SD Major forks

Main: https://github.com/CompVis/stable-diffusion

- https://github.com/basujindal/stable-diffusion 
  This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed.
- https://github.com/hlky/stable-diffusion
  - adds a bunch of features - GUI/webui, [textual inversion](https://textual-inversion.github.io/), [upscalers](https://github.com/hlky/stable-diffusion-webui/wiki/Upscalers), mask and crop, img2img editor, word seeds, prompt weighting
  - How to Fine-tune Stable Diffusion using Textual Inversion https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-textual-inversion-b995d7ecc095
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
- https://github.com/harubaru/waifu-diffusion
  - nicer GUI for img2img

SD Tooling

- SD's DreamStudio https://beta.dreamstudio.ai/dream
- Midjourney + SD: https://twitter.com/EMostaque/status/1561917541743841280
- [Nightcafe Studio](https://creator.nightcafe.studio/stable-diffusion-image-generator)
- misc
  - (super super raw dont try yet) https://github.com/breadthe/sd-buddy

## SD Model values


- How SD works https://twitter.com/johnowhitaker/status/1565710033463156739
- [Exploring 12 Million of the 2.3 Billion Images Used to Train Stable Diffusion’s Image Generator](https://waxy.org/2022/08/exploring-12-million-of-the-images-used-to-train-stable-diffusions-image-generator/)


## SD Results

### Img2Img

- A black and white photo of a young woman, studio lighting, realistic, Ilford HP5 400
  - https://twitter.com/TomLikesRobots/status/1566027217892671488
