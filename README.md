# prompt-eng

notes for prompt engineering

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Motivational Use Cases](#motivational-use-cases)
- [Top Prompt Engineering Reads](#top-prompt-engineering-reads)
- [Tooling](#tooling)
- [Communities](#communities)
- [Stable Diffusion](#stable-diffusion)
  - [SD Distros](#sd-distros)
  - [SD Major forks](#sd-major-forks)
  - [SD Prompt galleries and search engines](#sd-prompt-galleries-and-search-engines)
  - [SD Visual search](#sd-visual-search)
  - [SD Prompt generators](#sd-prompt-generators)
  - [Img2prompt - Reverse Prompt Engineering](#img2prompt---reverse-prompt-engineering)
  - [Explore Artists, styles, and modifiers](#explore-artists-styles-and-modifiers)
  - [SD Prompt Tools directories and guides](#sd-prompt-tools-directories-and-guides)
- [How SD Works - Internals and Studies](#how-sd-works---internals-and-studies)
- [SD Results](#sd-results)
  - [Img2Img](#img2img)
- [Hardware requirements](#hardware-requirements)
- [SD vs DallE vs MJ](#sd-vs-dalle-vs-mj)
- [Misc](#misc)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Motivational Use Cases

- images
  - https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts/
- video
  - img2img of famous movie scenes ([lalaland](https://twitter.com/TomLikesRobots/status/1565678995986911236))
  - virtual fashion ([karenxcheng](https://twitter.com/karenxcheng/status/1564626773001719813))
  - evolution of scenes ([xander](https://twitter.com/xsteenbrugge/status/1558508866463219712))
  - outpainting https://twitter.com/orbamsterdam/status/1568200010747068417?s=21&t=rliacnWOIjJMiS37s8qCCw
  - webUI img2img collaboration https://twitter.com/_akhaliq/status/1563582621757898752
  - image to video with rotation https://twitter.com/TomLikesRobots/status/1571096804539912192
  - "prompt paint" https://twitter.com/1littlecoder/status/1572573152974372864
  - music videos [video](https://www.youtube.com/watch?v=WJaxFbdjm8c), [colab](https://colab.research.google.com/github/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb)
  - direct text2video project
    - https://twitter.com/_akhaliq/status/1575546841533497344
    - https://makeavideo.studio/ - explorer https://webvid.datasette.io/webvid/videos
    - https://phenaki.video/
    - https://github.com/THUDM/CogVideo
    - https://imagen.research.google/video/
- text-to-3d https://twitter.com/_akhaliq/status/1575541930905243652
  -  https://dreamfusion3d.github.io/
  -  open source impl: https://github.com/ashawkey/stable-dreamfusion

## Top Prompt Engineering Reads

The more advanced GPT3 reads have been split out to https://github.com/sw-yx/prompt-eng/blob/main/GPT.md

- https://www.gwern.net/GPT-3#prompts-as-programming
- beginner
  - openAI prompt tutorial https://beta.openai.com/docs/quickstart/add-some-examples
  - DALLE2 prompt writing book http://dallery.gallery/wp-content/uploads/2022/07/The-DALL%C2%B7E-2-prompt-book-v1.02.pdf
  - https://medium.com/nerd-for-tech/prompt-engineering-the-career-of-future-2fb93f90f117
  - https://wiki.installgentoo.com/wiki/Stable_Diffusion overview
  - https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
  - https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts/
  - https://andymatuschak.org/prompts/
- Intermediate
  - DALLE2 asset generation + inpainting https://twitter.com/aifunhouse/status/1576202480936886273?s=20&t=5EXa1uYDPVa2SjZM-SxhCQ
  - suhail journey https://twitter.com/Suhail/status/1541276314485018625?s=20&t=X2MVKQKhDR28iz3VZEEO8w
  - composable diffusion - "AND" instead of "and" https://twitter.com/TomLikesRobots/status/1580293860902985728
  - quest for photorealism https://www.reddit.com/r/StableDiffusion/comments/x9zmjd/quest_for_ultimate_photorealism_part_2_colors/
    - https://medium.com/merzazine/prompt-design-for-dall-e-photorealism-emulating-reality-6f478df6f186
  - settings tweaking https://www.reddit.com/r/StableDiffusion/comments/x3k79h/the_feeling_of_discovery_sd_is_like_a_great_proc/
    - seed selection https://www.reddit.com/r/StableDiffusion/comments/x8szj9/tutorial_seed_selection_and_the_impact_on_your/
    - minor parameter parameter difference study (steps, clamp_max, ETA, cutn_batches, etc) https://twitter.com/KyrickYoung/status/1500196286930292742
- Advanced
  - nothing yet
- https://creator.nightcafe.studio/vqgan-clip-keyword-modifier-comparison VQGAN+CLIP Keyword Modifier Comparison
We compared 126 keyword modifiers with the same prompt and initial image. These are the results.
  - https://creator.nightcafe.studio/collection/8dMYgKm1eVXG7z9pV23W
- Google released PartiPrompts as a benchmark: https://parti.research.google/ "PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as part of this work. P2 can be used to measure model capabilities across various categories and challenge aspects."
- Video tutorials
  - Pixel art https://www.youtube.com/watch?v=UvJkQPtr-8s&feature=youtu.be
- Misc
  - StabilityAI CIO perspective https://danieljeffries.substack.com/p/the-turning-point-for-truly-open?sd=pf
  - https://github.com/awesome-stable-diffusion/awesome-stable-diffusion

## Tooling

- Prompt Generators: 
  - https://huggingface.co/succinctly/text2image-prompt-generator
    - This is a GPT-2 model fine-tuned on the succinctly/midjourney-prompts dataset, which contains 250k text prompts that users issued to the Midjourney text-to-image service over a month period. This prompt generator can be used to auto-complete prompts for any text-to-image model (including the DALL·E family)
  - Prompt Parrot https://colab.research.google.com/drive/1GtyVgVCwnDfRvfsHbeU0AlG-SgQn1p8e?usp=sharing
    - This notebook is designed to train language model on a list of your prompts,generate prompts in your style, and synthesize wonderful surreal images! ✨
  - https://twitter.com/stuhlmueller/status/1575187860063285248
    - The Interactive Composition Explorer (ICE), a Python library for writing and debugging compositional language model programs https://github.com/oughtinc/ice
  - The Factored Cognition Primer, a tutorial that shows using examples how to write such programs https://primer.ought.org
  - Prompt Explorer
    - https://twitter.com/fabianstelzer/status/1575088140234428416
    - https://docs.google.com/spreadsheets/d/1oi0fwTNuJu5EYM2DIndyk0KeAY8tL6-Qd1BozFb9Zls/edit#gid=1567267935 
  - Prompt generator https://www.aiprompt.io/

misc

- [Edsynth](https://www.youtube.com/watch?v=eghGQtQhY38) and [DAIN](https://twitter.com/karenxcheng/status/1564635828436885504) for coherence
- [FILM: Frame Interpolation for Large Motion](https://film-net.github.io/) ([github](https://github.com/google-research/frame-interpolation))
- [Depth Mapping](https://github.com/compphoto/BoostingMonocularDepth)
  - examples: https://twitter.com/TomLikesRobots/status/1566152352117161990
- Art program plugins
  - Krita: https://github.com/nousr/koi
  - GIMP https://80.lv/articles/a-new-stable-diffusion-plug-in-for-gimp-krita/
  - Photoshop: https://old.reddit.com/r/StableDiffusion/comments/wyduk1/show_rstablediffusion_integrating_sd_in_photoshop/
    - download: https://twitter.com/cantrell/status/1574432458501677058
    - https://www.getalpaca.io/
    - demo: https://www.youtube.com/watch?v=t_4Y6SUs1cI
    - tutorial https://odysee.com/@MaxChromaColor:2/how-to-install-the-free-stable-diffusion:1
  - Figma: https://twitter.com/RemitNotPaucity/status/1562319004563173376?s=20&t=fPSI5JhLzkuZLFB7fntzoA
  - collage tool https://twitter.com/genekogan/status/1555184488606564353
- Papers
  - 2015: [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf) founding paper of diffusion models
  - Textual Inversion: https://arxiv.org/abs/2208.01618 (impl: https://github.com/rinongal/textual_inversion)
  - 2017: Attention is all you need
  - https://dreambooth.github.io/
    - productized as dreambooth https://twitter.com/psuraj28/status/1575123562435956740
    - https://github.com/JoePenna/Dreambooth-Stable-Diffusion
  - [very good BLOOM model overview](https://www.youtube.com/watch?v=3EjtHs_lXnk)

## Communities

- StableDiffusion Discord https://discord.com/invite/stablediffusion
- https://reddit.com/r/stableDiffusion
- Akhaliq Discord: https://discord.gg/nYqfg4gnBt
- Deforum Discord https://discord.gg/upmXXsrwZc
- Lexica Discord https://discord.com/invite/bMHBjJ9wRh
- Midjourney's discord
- https://stablehorde.net/

## Stable Diffusion

stable diffusion specific notes

Required reading:
- param intuitionhttps://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
- CLI commands https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/#script-options

### SD Distros

- **Installer Distros**: Programs that bundle Stable Diffusion in an installable program, no separate setup and the least amount of git/technical skill needed, usually bundling one or more UI
  - [Diffusion Bee](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui): Diffusion Bee is the easiest way to run Stable Diffusion locally on your M1 Mac. Comes with a one-click installer. No dependencies or technical knowledge needed.
  - https://www.charl-e.com/: Stable Diffusion on your Mac in 1 click. ([tweet](https://twitter.com/charliebholtz/status/1571138577744138240))
  - https://github.com/cmdr2/stable-diffusion-ui: Easiest 1-click way to install and use Stable Diffusion on your own computer. Provides a browser UI for generating images from text prompts and images. Just enter your text prompt, and see the generated image. (Linux, Windows, no Mac)
  - https://nmkd.itch.io/t2i-gui: A basic (for now) Windows 10/11 64-bit GUI to run Stable Diffusion, a machine learning toolkit to generate images from text, locally on your own hardware. As of right now, this program only works on Nvidia GPUs! AMD GPUs are not supported. In the future this might change. 
  - [imaginAIry 🤖🧠](https://github.com/brycedrennan/imaginAIry): Pythonic generation of stable diffusion images with just `pip install imaginairy`. "just works" on Linux and macOS(M1) (and maybe windows). Memory efficiency improvements, prompt-based editing, face enhancement, upscaling, tiled images, img2img, prompt matrices, prompt variables, BLIP image captions, comes with dockerfile/colab.  Has unit tests.
  - [Fictiverse/Windows-GUI](https://github.com/Fictiverse/StableDiffusion-Windows-GUI): A windows interface for stable diffusion
  - https://github.com/razzorblade/stable-diffusion-gui: dormant now.
- **Web Distros**
  - https://www.mage.space/
  - https://dreamlike.art/ has img2img
  - https://inpainter.vercel.app/paint for inpainting
  - https://promptart.labml.ai/feed
  - https://www.strmr.com/ dreambooth tuning for $3
  - https://www.findanything.app browser extension that adds SD predictions alongside Google search
  - https://www.drawanything.app 
- **Twitter Bots**
  - https://twitter.com/diffusionbot
  - https://twitter.com/m1guelpf/status/1569487042345861121
- **Windows "retard guides"**
  - https://rentry.org/voldy
  - https://rentry.org/GUItard

### SD Major forks

Main Stable Diffusion repo: https://github.com/CompVis/stable-diffusion

| Name/Link 	| Stars 	| Description 	|
|---	|---	|---	|
| [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 	| 9700 	| The most well known fork. features: https://github.com/AUTOMATIC1111/stable-diffusion-webui#features launch announcement https://www.reddit.com/r/StableDiffusion/comments/x28a76/stable_diffusion_web_ui/. M1 mac instructions https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon 	|
| [Disco Diffusion](https://github.com/alembics/disco-diffusion) 	| 5600 	| A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations. 	|
| [sd-webui](https://github.com/sd-webui/stable-diffusion-webui) (formerly hlky fork) 	| 5100 	| A fully-integrated and easy way to work with Stable Diffusion right from a browser window. Long list of UI and SD features (incl textual inversion, alternative samplers, prompt matrix): https://github.com/sd-webui/stable-diffusion-webui#project-features 	|
| [InvokeAI](https://github.com/invoke-ai/InvokeAI) (formerly lstein fork) 	| 3400 	| This version of Stable Diffusion features a slick WebGUI, an interactive command-line script that combines text2img and img2img functionality in a "dream bot" style interface, and multiple features and other enhancements. It runs on Windows, Mac and Linux machines, with GPU cards with as little as 4 GB of RAM.  	|
| [XavierXiao/Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) 	| 2400 	| Implementation of Dreambooth (https://arxiv.org/abs/2208.12242) with Stable Diffusion. Dockerized: https://github.com/smy20011/dreambooth-docker 	|
| [Basujindal: Optimized Stable Diffusion](https://github.com/basujindal/stable-diffusion) 	| 2100 	| This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed. img2img and txt2img and inpainting under 2.4GB VRAM  	|
| [stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity) 	| 1900 	| Outpainting with Stable Diffusion on an infinite canvas. This project mainly works as a proof of concept.  	|
| [Waifu Diffusion](https://github.com/harubaru/waifu-diffusion) ([huggingface](https://huggingface.co/hakurei/waifu-diffusion), [replicate](https://replicate.com/cjwbw/waifu-diffusion)) 	| 1100 	| stable diffusion finetuned on weeb stuff. "A model trained on danbooru (anime/manga drawing site with also lewds and nsfw on it) over 56k images.Produces FAR BETTER results if you're interested in getting manga and anime stuff out of stable diffusion." 	|
| [AbdBarho/stable-diffusion-webui-docker](https://github.com/AbdBarho/stable-diffusion-webui-docker) 	| 929 	| Easy Docker setup for Stable Diffusion with both Automatic1111 and hlky UI included. HOWEVER - no mac support yet https://github.com/AbdBarho/stable-diffusion-webui-docker/issues/35 	|
| [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) 	| 753 	|  +25-50% speed increase + memory efficient + DreamBooth 	|
| [imaginAIry 🤖🧠](https://github.com/brycedrennan/imaginAIry) | 639 | Pythonic generation of stable diffusion images with just `pip install imaginairy`. "just works" on Linux and macOS(M1) (and maybe windows). Memory efficiency improvements, prompt-based editing, face enhancement, upscaling, tiled images, img2img, prompt matrices, prompt variables, BLIP image captions, comes with dockerfile/colab.  Has unit tests. |
| [neonsecret/stable-diffusion](https://github.com/neonsecret/stable-diffusion) 	| 546 	| This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed. Also I invented the sliced atttention technique, which allows to push the model's abilities even further. It works by automatically determining the slice size from your vram and image size and then allocating it one by one accordingly. You can practically generate any image size, it just depends on the generation speed you are willing to sacrifice. 	|
| [Deforum Stable Diffusion](https://github.com/deforum/stable-diffusion) 	| 347 	| Animating prompts with stable diffusion. replicate demo: https://replicate.com/deforum/deforum_stable_diffusion 	|
| [Doggettx/stable-diffusion](https://github.com/Doggettx/stable-diffusion) 	| 137 	| Allows to use resolutions that require up to 64x more VRAM than possible on the default CompVis build. 	|

#### SD in Other languages

- Chinese: https://twitter.com/_akhaliq/status/1572580845785083906
- Japanese: https://twitter.com/_akhaliq/status/1571977273489739781
  - https://huggingface.co/blog/japanese-stable-diffusion
  
#### Other Lists of Forks

- https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/
- https://www.reddit.com/r/StableDiffusion/comments/xcclmf/comment/io6u03s/?utm_source=reddit&utm_medium=web2x&context=3
- https://techgaun.github.io/active-forks/index.html#CompVis/stable-diffusion

Dormant projects, for historical/research interest:

- https://colab.research.google.com/drive/1AfAmwLMd_Vx33O9IwY2TmO9wKZ8ABRRa		
- https://colab.research.google.com/drive/1kw3egmSn-KgWsikYvOMjJkVDsPLjEMzl		
- [bfirsh/stable-diffusion](https://github.com/bfirsh/stable-diffusion)	No longer actively maintained byt was the first to work on M1 Macs - [blog](https://replicate.com/blog/run-stable-diffusion-on-m1-mac), [tweet](https://twitter.com/levelsio/status/1565731907664478209), can also look at `environment-mac.yaml` from https://github.com/fragmede/stable-diffusion/blob/mps_consistent_seed/environment-mac.yaml

#### Misc SD UI's

UI's that dont come with their own SD distro, just shelling out to one

| UI Name/Link 	| Stars 	| Self-Description 	|
|---	|---	|---	|
| [ahrm/UnstableFusion](https://github.com/ahrm/UnstableFusion) 	| 815 	| UnstableFusion is a desktop frontend for Stable Diffusion which combines image generation, inpainting, img2img and other image editing operation into a seamless workflow.  https://www.youtube.com/watch?v=XLOhizAnSfQ&t=1s 	|
| [breadthe/sd-buddy](https://github.com/breadthe/sd-buddy/) 	| 165 	| Companion desktop app for the self-hosted M1 Mac version of Stable Diffusion, with Svelte and Tauri 	|
| [leszekhanusz/diffusion-ui](https://github.com/leszekhanusz/diffusion-ui) 	| 65 	| This is a web interface frontend for the generation of images using diffusion models.<br><br>The goal is to provide an interface to online and offline backends doing image generation and inpainting like Stable Diffusion. 	|
| [GenerationQ](https://github.com/westoncb/generation-q) 	| 21 	| GenerationQ (for "image generation queue") is a cross-platform desktop application (screens below) designed to provide a general purpose GUI for generating images via text2img and img2img models. Its primary target is Stable Diffusion but since there is such a variety of forked programs with their own particularities, the UI for configuring image generation tasks is designed to be generic enough to accommodate just about any script (even non-SD models). 	|


### SD Prompt galleries and search engines

- 🌟 [Lexica](https://lexica.art/): Content-based search powered by OpenAI's CLIP model. **Seed**, CFG, Dimensions.
- https://synesthetic.ai/ SD focused
- https://visualise.ai/ Create and share image prompts. DALL-E, Midjourney, Stable Diffusion
- https://nyx.gallery/
- [OpenArt](https://openart.ai/discovery?dataSource=sd): Content-based search powered by OpenAI's CLIP model. Favorites.
- [PromptHero](https://prompthero.com/): [Random wall](https://prompthero.com/random). **Seed**, CFG, Dimensions, Steps. Favorites.
- [Libraire](https://libraire.ai/): **Seed**, CFG, Dimensions, Steps.
- [Krea](https://www.krea.ai/): modifiers focused UI. Favorites. Gives prompt suggestions and allows to create prompts over Stable diffusion, Waifu Diffusion and Disco diffusion. Really quick and useful
- [Avyn](http://avyn.com/): Search engine and generator.
- [Pinegraph](https://pinegraph.com/): [discover](https://pinegraph.com/discover), [create](https://pinegraph.com/create) and edit with Stable/Disco/Waifu diffusion models.
- [Phraser](https://phraser.tech/compare): text and image search.
- https://arthub.ai/
- https://pagebrain.ai/promptsearch/
- https://avyn.com/
- https://dallery.gallery/
- [The Ai Art:](https://www.the-ai-art.com/modifiers) **gallery** for modifiers.
- [urania.ai](https://www.urania.ai/top-sd-artists): Top 500 Artists **gallery**, sorted by [image count](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/artists?_sort_desc=image_counts). With modifiers/styles.
- [Generrated](https://generrated.com/): DALL•E 2 table **gallery** sorted by [visual arts media](https://en.wikipedia.org/wiki/Category:Visual_arts_media).
- [Artist Studies by @remi_durant](https://remidurant.com/artists/): **gallery** and Search.
- [CLIP Ranked Artists](https://f000.backblazeb2.com/file/clip-artists/index.html): **gallery** sorted by weight/strength.
- https://promptbase.com/ Selling prompts that produce desirable results
- https://publicprompts.art/ very basic/limited but some good prompts. promptbase competitor

### SD Visual search

- [Lexica](https://lexica.art/?q=): enter an image URL in the search bar. Or next to q=. [Example](https://lexica.art/?q=https%3A%2F%2Fi.imgur.com%2FNyURMpx.jpeg)
- [Phraser](https://phraser.tech/compare): image icon at the right.
- [same.energy](https://same.energy/)
- [Yandex](https://yandex.com/images/), [Bing](https://www.bing.com/images/feed), [Google](https://www.google.com/imghp), [Tineye](https://www.tineye.com/), [iqdb](https://iqdb.org/): reverse and similar image search engines.
- [Pinterest](https://www.pinterest.com/search/)
- [dessant/search-by-image](https://github.com/dessant/search-by-image): Open-source browser extension for reverse image search.

### SD Prompt generators

- [promptoMANIA](https://promptomania.com/prompt-builder/): **Visual** modifiers. Great selection. With weight setting.
- [Phase.art](https://www.phase.art/): **Visual** modifiers. SD [Generator and share](https://www.phase.art/images/cl826cjsb000509mlwqbael1i).
- [Phraser](https://phraser.tech/): **Visual** modifiers.
- [AI Text Prompt Generator](https://aitextpromptgenerator.com/)
- [Dynamic Prompt generator](https://rexwang8.github.io/resource/ai/generator)
- [succinctly/text2image](https://huggingface.co/succinctly/text2image-prompt-generator): GPT-2 Midjourney trained text completion.
- [Prompt Parrot colab](https://colab.research.google.com/drive/1GtyVgVCwnDfRvfsHbeU0AlG-SgQn1p8e?usp=sharing): Train and generate prompts.
- [cmdr2](https://github.com/cmdr2/stable-diffusion-ui): 1-click SD installation with image modifiers selection.

### Img2prompt - Reverse Prompt Engineering

- [img2prompt](https://replicate.com/methexis-inc/img2prompt) Replicate by [methexis-inc](https://replicate.com/methexis-inc): Optimized for SD (clip ViT-L/14).
- [CLIP Interrogator](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) by [@pharmapsychotic](https://twitter.com/pharmapsychotic): select ViTL14 CLIP model.
- [CLIP Artist Evaluator colab](https://colab.research.google.com/github/lowfuel/CLIP_artists/blob/main/CLIP_Evaluator.ipynb)
- [BLIP](https://huggingface.co/spaces/Salesforce/BLIP)

### Explore Artists, styles, and modifiers

See https://github.com/sw-yx/prompt-eng/blob/main/PROMPTS.md for more details and notes

- [Artist Style Studies](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d) & [Modifier Studies](https://www.notion.so/2b07d3195d5948c6a7e5836f9d535592) by [parrot zone](https://www.notion.so/74a5c04d4feb4f12b52a41fc8750b205): **[Gallery](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d)**, [Style](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d), [Spreadsheet](https://docs.google.com/spreadsheets/d/14xTqtuV3BuKDNhLotB_d1aFlBGnDJOY0BRXJ8-86GpA/edit#gid=0)
- [Clip retrieval](https://knn5.laion.ai/): search [laion-5b](https://laion.ai/blog/laion-5b/) dataset.
- [Datasette](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls): [image search](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/images); image-count sort by [artist](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/artists?_sort_desc=image_counts), [celebrities](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/celebrities?_sort_desc=image_counts), [characters](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/characters?_sort_desc=image_counts), [domain](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/domain?_sort_desc=image_counts)
- [Visual](https://en.wikipedia.org/wiki/Category:Visual_arts) [arts](https://en.wikipedia.org/wiki/Category:The_arts): [media](https://en.wikipedia.org/wiki/Category:Visual_arts_media) [list](https://en.wikipedia.org/wiki/List_of_art_media), [related](https://en.wikipedia.org/wiki/Category:Arts-related_lists); [Artists](https://en.wikipedia.org/wiki/Category:Artists) [list](https://en.wikipedia.org/wiki/Category:Lists_of_artists) by [genre](https://en.wikipedia.org/wiki/Category:Artists_by_genre), [medium](https://en.wikipedia.org/wiki/Category:Artists_by_medium); [Portal](https://en.wikipedia.org/wiki/Portal:The_arts)

### SD Prompt Tools directories and guides

- Useful Prompt Engineering tools and resources https://np.reddit.com/r/StableDiffusion/comments/xcrm4d/useful_prompt_engineering_tools_and_resources/
- [Tools and Resources for AI Art](https://pharmapsychotic.com/tools.html) by [pharmapsychotic](https://www.reddit.com/user/pharmapsychosis)
- [Akashic Records](https://github.com/Maks-s/sd-akashic#prompts-toc)
- [Awesome Stable-Diffusion](https://github.com/Maks-s/sd-akashic#prompts-toc)

#### SD Tooling

- AI Dreamer iOS/macOS app https://apps.apple.com/us/app/ai-dreamer/id1608856807
- SD's DreamStudio https://beta.dreamstudio.ai/dream
- Stable Worlds: [colab](https://colab.research.google.com/drive/1RXRrkKUnpNiPCxTJg0Imq7sIM8ltYFz2?usp=sharing) for 3d stitched worlds via StableDiffusion https://twitter.com/NaxAlpha/status/1578685845099290624
- Midjourney + SD: https://twitter.com/EMostaque/status/1561917541743841280
- [Nightcafe Studio](https://creator.nightcafe.studio/stable-diffusion-image-generator)
- misc
  - (super super raw dont try yet) https://github.com/breadthe/sd-buddy


## How SD Works - Internals and Studies

- How SD works
  - https://huggingface.co/blog/stable_diffusion
  - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
  - https://twitter.com/johnowhitaker/status/1565710033463156739
  - https://twitter.com/ai__pub/status/1561362542487695360
  - https://twitter.com/JayAlammar/status/1572297768693006337
    - https://jalammar.github.io/illustrated-stable-diffusion/
  - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
  - inside https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/#wait-how-does-this-even-work
- Samplers studies
  - https://twitter.com/iScienceLuvr/status/1564847717066559488
- [Disco Diffusion Illustrated Settings](https://www.notion.so/cd4badf06e08440c99d8a93d4cd39f51)
- [Understanding MidJourney (and SD) through teapots.](https://rexwang8.github.io/resource/ai/teapot)
- [A Traveler’s Guide to the Latent Space](https://www.notion.so/85efba7e5e6a40e5bd3cae980f30235f)
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

## Misc

- Imagen
  - https://www.youtube.com/watch?v=R_f-v6prMqI
- Whisper
  - https://huggingface.co/spaces/sensahin/YouWhisper YouWhisper converts Youtube videos to text using openai/whisper.
  - https://twitter.com/jeffistyping/status/1573145140205846528 youtube whipserer
  - multilingual subtitles https://twitter.com/1littlecoder/status/1573030143848722433
  - video subtitles https://twitter.com/m1guelpf/status/1574929980207034375
  - you can join whisper to stable diffusion for reasons https://twitter.com/fffiloni/status/1573733520765247488/photo/1
  - known problems https://twitter.com/lunixbochs/status/1574848899897884672 (edge case with catastrophic failures)
- textually guided audio https://twitter.com/FelixKreuk/status/1575846953333579776
- Codegen
  - CodegeeX https://twitter.com/thukeg/status/1572218413694726144
  - https://github.com/salesforce/CodeGen https://joel.tools/codegen/
- pdf to structured data https://www.impira.com/blog/hey-machine-whats-my-invoice-total
- text to Human Motion diffusion https://twitter.com/GuyTvt/status/1577947409551851520
  - abs: https://arxiv.org/abs/2209.14916 
  - project page: https://guytevet.github.io/mdm-page/
