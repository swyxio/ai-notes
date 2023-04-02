
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [good reads](#good-reads)
- [SD vs DallE vs MJ](#sd-vs-dalle-vs-mj)
  - [DallE](#dalle)
- [Tooling](#tooling)
- [Products](#products)
- [Stable Diffusion prompts](#stable-diffusion-prompts)
  - [SD v2 prompts](#sd-v2-prompts)
  - [SD 1.4 vs 1.5 comparisons](#sd-14-vs-15-comparisons)
- [Distilled Stable Diffusion](#distilled-stable-diffusion)
- [SD2 vs SD1 user notes](#sd2-vs-sd1-user-notes)
- [Hardware requirements](#hardware-requirements)
- [Stable Diffusion](#stable-diffusion)
  - [SD Distros](#sd-distros)
  - [SD Major forks](#sd-major-forks)
  - [SD Prompt galleries and search engines](#sd-prompt-galleries-and-search-engines)
  - [SD Visual search](#sd-visual-search)
  - [SD Prompt generators](#sd-prompt-generators)
  - [Img2prompt - Reverse Prompt Engineering](#img2prompt---reverse-prompt-engineering)
  - [Explore Artists, styles, and modifiers](#explore-artists-styles-and-modifiers)
  - [SD Prompt Tools directories and guides](#sd-prompt-tools-directories-and-guides)
  - [Finetuning/Dreambooth](#finetuningdreambooth)
- [How SD Works - Internals and Studies](#how-sd-works---internals-and-studies)
- [SD Results](#sd-results)
  - [Img2Img](#img2img)
- [Extremely detailed prompt examples](#extremely-detailed-prompt-examples)
  - [Solving Hands](#solving-hands)
- [Midjourney prompts](#midjourney-prompts)
- [Misc](#misc)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## good reads

- Ten Years of Image Synthesis https://zentralwerkstatt.org/blog/ten-years-of-image-synthesis
	- 2014-2017 https://twitter.com/swyx/status/1049412858755264512
	- 2014-2022 https://twitter.com/c_valenzuelab/status/1562579547404455936
	- wolfenstein 1992 vs 2014 https://twitter.com/kevinroose/status/1557815883837255680
	- april 2022 dalle 2
	- july 2022 craiyon/dailee mini
	- aug 2022 stable diffusion
	- getty, shutterstock, canva incorporated
	- midjourney progression in 2022 https://twitter.com/lopp/status/1595846677591904257
	- eDiffi
- Vision Transformers (ViT) Explained https://www.pinecone.io/learn/vision-transformers/
	- team at Google Brain introduced [vision transformers](https://arxiv.org/abs/2010.11929?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8HbXG-ZkwAj82Nv49uUrBwOHz4zUj3mkyjIfEd5lU7h3JHZR0pEG5OpkUCPPqwWvqMbjWl) (ViTs) in 2020, and the architecture has undergone nonstop refinement since then. The latest efforts adapt ViTs to new tasks and address their shortcomings.
	-   ViTs learn best from immense quantities of data, so researchers at Meta and Sorbonne University concentrated on [improving performance on datasets of (merely) millions of examples](https://www.deeplearning.ai/the-batch/a-formula-for-training-vision-transformers/). They boosted performance using transformer-specific adaptations of established procedures such as data augmentation and model regularization.
	-   Researchers at Inha University modified two key components to make ViTs [more like convolutional neural networks](https://www.deeplearning.ai/the-batch/less-data-for-vision-transformers/). First, they divided images into patches with more overlap. Second, they modified self-attention to focus on a patch's neighbors rather than on the patch itself, and enabled it to learn whether to weigh neighboring patches more evenly or more selectively. These modifications brought a significant boost in accuracy.
	-   Researchers at the Indian Institute of Technology Bombay [outfitted ViTs with convolutional layers](https://www.deeplearning.ai/the-batch/upgrade-for-vision-transformers/). Convolution brings benefits like local processing of pixels and smaller memory footprints due to weight sharing. With respect to accuracy and speed, their convolutional ViT outperformed the usual version as well as runtime optimizations of transformers such as Performer, Nyströformer, and Linear Transformer. Other teams took [similar](https://arxiv.org/abs/2201.09792?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8HbXG-ZkwAj82Nv49uUrBwOHz4zUj3mkyjIfEd5lU7h3JHZR0pEG5OpkUCPPqwWvqMbjWl) [approaches](https://arxiv.org/abs/2202.06709?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8HbXG-ZkwAj82Nv49uUrBwOHz4zUj3mkyjIfEd5lU7h3JHZR0pEG5OpkUCPPqwWvqMbjWl).
	- more from fchollet: https://keras.io/examples/vision/probing_vits/
- CLIP (_Contrastive Language–Image Pre-training_) https://openai.com/blog/clip/
	- https://ml.berkeley.edu/blog/posts/clip-art/
		- jan 2021 
			- On January 5th 2021, OpenAI released the model-weights and code for [CLIP](https://openai.com/blog/clip/): a model trained to determine which caption from a set of captions best fits with a given image.
			- The Big Sleep: a CLIP based text-to-image technique ([source](https://twitter.com/advadnoun/status/1351038053033406468))
		- may 2021: [the unreal engine trick](https://ml.berkeley.edu/blog/posts/clip-art/#the-joys-of-prompt-programming-the-unreal-engine-trick)
	- CLIPSeg https://huggingface.co/docs/transformers/main/en/model_doc/clipseg (for Image segmentation)
	- Queryable - CLIP on iphone photos https://news.ycombinator.com/item?id=34686947
		- on website https://paulw.tokyo/post/real-time-semantic-search-demo/
	- beating CLIP # with 100x less data and compute https://www.unum.cloud/blog/2023-02-20-efficient-multimodality
	- https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html
- Stable Diffusion
	- https://stability.ai/blog/stable-diffusion-v2-release
		- _New Text-to-Image Diffusion Models_
		- _Super-resolution Upscaler Diffusion Models_
		- _Depth-to-Image Diffusion Model_
		- _Updated Inpainting Diffusion Model_
		- https://news.ycombinator.com/item?id=33726816
			- Seems the structure of UNet hasn't changed other than the text encoder input (768 to 1024). The biggest change is on the text encoder, switched from ViT-L14 to ViT-H14 and fine-tuned based on [https://arxiv.org/pdf/2109.01903.pdf](https://arxiv.org/pdf/2109.01903.pdf).
			- the dataset it's trained on is ~240TB (5 billion pairs of text to 512x512 image.) and Stability has over ~4000 Nvidia A100
		- Runway vs Stable Diffusion drama https://www.forbes.com/sites/kenrickcai/2022/12/05/runway-ml-series-c-funding-500-million-valuation/
	- https://stability.ai/blog/stablediffusion2-1-release7-dec-2022
		- Better people and less restrictions than v2.0
		- Nonstandard resolutions
		- Dreamstudio with negative prompts and weights
		- https://old.reddit.com/r/StableDiffusion/comments/zf21db/stable_diffusion_21_announcement/
	- Stability 2022 recap https://twitter.com/StableDiffusion/status/1608661612776550401
	- https://stablediffusionlitigation.com
- important papers
	- 2019 Razavi, Oord, Vinyals, [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)
	- 2020 Esser, Rombach, Ommer, [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
		- ([summary](https://twitter.com/sedielem/status/1339929984836788228)) To synthesise realistic megapixel images, learn a high-level discrete representation with a conditional GAN, then train a transformer on top. Likelihood-based models like transformers do better at capturing diversity compared to GANs, but tend to get lost in the details. Likelihood is mode-covering; not mode-seeking, like adversarial losses are. By measuring the likelihood in a space where texture details have been abstracted away, the transformer is forced to capture larger-scale structure, and we get great compositions as a result. Replacing the VQ-VAE with a VQ-GAN enables more aggressive downsampling.
	- 2021 Dhariwal & Nichol, [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
	- 2021 Nichol et al, [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)

## SD vs DallE vs MJ

DallE banned so SD https://twitter.com/almost_digital/status/1556216820788609025?s=20&t=GCU5prherJvKebRrv9urdw

[![https://i.redd.it/fqgv82ihav9a1.png](https://i.redd.it/fqgv82ihav9a1.png)](https://www.reddit.com/r/dalle2/comments/102eov5/who_did_it_better_dalle_2_midjourney_and_stable/?s=8) but keep in mind that Dalle2 [doesnt respond well](https://www.reddit.com/r/dalle2/comments/waax7p/realistic_and_photorealistic_keywords_give/) to "photorealistic"

another comparison https://www.reddit.com/r/StableDiffusion/comments/zevuw2/a_simple_comparison_between_sd_15_20_21_and/

comparisons with other models https://www.reddit.com/r/StableDiffusion/comments/zlvrl6/i_tried_various_models_with_the_same_settings/

Lexica Aperture - finetuned version of SD https://lexica.art/aperture
	- fast
	- focused on photorealistic portraits and landscapes
	- negative prompting
	- dimensions

## midjourney

- midjourney company is 10 people and 40 moderators https://www.washingtonpost.com/technology/2023/03/30/midjourney-ai-image-generation-rules/
-   [Advanced guide to writing prompts for MidJourney](https://medium.com/mlearning-ai/an-advanced-guide-to-writing-prompts-for-midjourney-text-to-image-aa12a1e33b6) 
-   [Aspect ratio prompts](https://graphicsgurl.com/midjourney-aspect-ratio/#:~:text=MidJourney's%20default%20size%20is%20square,ratios%20%E2%80%93%20this%20is%20the%20original)

### Midjourney v5

- [rave at Hogwarts summer 1998](https://twitter.com/spacecasetay/status/1638212304683532288)
- midjourney prompting with gpt4 https://twitter.com/nickfloats/status/1638679555107094528
- fashion liv boeree prompt https://twitter.com/nickfloats/status/1639076580419928068
- extremely photorealistic, lots of interesting examples https://twitter.com/bilawalsidhu/status/1639688267695112194



nice trick to mix images https://twitter.com/javilopen/status/1613107083959738369

"midjourney style" - just feed "prompt" to it https://twitter.com/rainisto/status/1606221760189317122

or emojis: https://twitter.com/LinusEkenstam/status/1616841985599365120

### DallE

- dallery gallery + prompt book https://news.ycombinator.com/item?id=32322329

DallE vs Imagen vs Parti  architecture
- https://twitter.com/savvyRL/status/1540555792331378688

### Runway Gen-1/2

usage example https://twitter.com/nickfloats/status/1639709828603084801?s=20

## Tooling

- Prompt Generators: 
  - https://huggingface.co/succinctly/text2image-prompt-generator
    - This is a GPT-2 model fine-tuned on the succinctly/midjourney-prompts dataset, which contains 250k text prompts that users issued to the Midjourney text-to-image service over a month period. This prompt generator can be used to auto-complete prompts for any text-to-image model (including the DALL·E family)
  - Prompt Parrot https://colab.research.google.com/drive/1GtyVgVCwnDfRvfsHbeU0AlG-SgQn1p8e?usp=sharing
    - This notebook is designed to train language model on a list of your prompts,generate prompts in your style, and synthesize wonderful surreal images! ✨
    - https://twitter.com/KyrickYoung/status/1563962142633648129
    - https://github.com/kyrick/cog-prompt-parrot
  - https://twitter.com/stuhlmueller/status/1575187860063285248
    - The Interactive Composition Explorer (ICE), a Python library for writing and debugging compositional language model programs https://github.com/oughtinc/ice
  - The Factored Cognition Primer, a tutorial that shows using examples how to write such programs https://primer.ought.org
  - Prompt Explorer
    - https://twitter.com/fabianstelzer/status/1575088140234428416
    - https://docs.google.com/spreadsheets/d/1oi0fwTNuJu5EYM2DIndyk0KeAY8tL6-Qd1BozFb9Zls/edit#gid=1567267935 
  - Prompt generator https://www.aiprompt.io/
- Stable Diffusion Interpolation
  - https://colab.research.google.com/drive/1EHZtFjQoRr-bns1It5mTcOVyZzZD9bBc?usp=sharing
  - This notebook generates neat interpolations between two different prompts with Stable Diffusion.
- Easy Diffusion by WASasquatch
  - This super nifty notebook has tons of features, such as image upscaling and processing, interrogation with CLIP, and more! (depth output for 3D Facebook images, or post processing such as Depth of Field.)
  - https://colab.research.google.com/github/WASasquatch/easydiffusion/blob/main/Stability_AI_Easy_Diffusion.ipynb
- Craiyon + Stable Diffusion https://twitter.com/GeeveGeorge/status/1567130529392373761
- Breadboard: https://www.reddit.com/r/StableDiffusion/comments/102ca1u/breadboard_a_stablediffusion_browser_version_010/
  -  a browser for effortlessly searching and managing all your Stablediffusion generated files.
    1. Full fledged browser UI: You can literally “surf” your local Stablediffusion generated files, home, back, forward buttons, search bar, and even bookmarks.
    2. Tagging: You can organize your files into tags, making it easy to filter them. Tags can be used to filter files in addition to prompt text searches.
    3. Bookmarking: You can now bookmark files. And you can bookmark search queries and tags. The UX is very similar to ordinary web browsers, where you simply click a star or a heart to favorite items.
    4. Realtime Notification: Get realtime notifications on all the Stablediffusion generated files.

Misc

- [prompt-engine](https://github.com/microsoft/prompt-engine): From Microsoft, NPM utility library for creating and maintaining prompts for LLMs
- [Edsynth](https://www.youtube.com/watch?v=eghGQtQhY38) and [DAIN](https://twitter.com/karenxcheng/status/1564635828436885504) for coherence
- [FILM: Frame Interpolation for Large Motion](https://film-net.github.io/) ([github](https://github.com/google-research/frame-interpolation))
- [Depth Mapping](https://github.com/compphoto/BoostingMonocularDepth)
  - examples: https://twitter.com/TomLikesRobots/status/1566152352117161990
- Art program plugins
  - Krita: https://github.com/nousr/koi
  - GIMP https://80.lv/articles/a-new-stable-diffusion-plug-in-for-gimp-krita/
  - Photoshop: https://old.reddit.com/r/StableDiffusion/comments/wyduk1/show_rstablediffusion_integrating_sd_in_photoshop/
	  - https://github.com/isekaidev/stable.art
	  - https://www.flyingdog.de/sd/
    - download: https://twitter.com/cantrell/status/1574432458501677058
    - https://www.getalpaca.io/
    - demo: https://www.youtube.com/watch?v=t_4Y6SUs1cI and https://twitter.com/cantrell/status/1582086537163919360
    - tutorial https://odysee.com/@MaxChromaColor:2/how-to-install-the-free-stable-diffusion:1
    - Photoshop with A1111 https://www.reddit.com/r/StableDiffusion/comments/zrdk60/great_news_automatic1111_photoshop_stable/
  - Figma: https://twitter.com/RemitNotPaucity/status/1562319004563173376?s=20&t=fPSI5JhLzkuZLFB7fntzoA
  - collage tool https://twitter.com/genekogan/status/1555184488606564353
- Papers
  - 2015: [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf) founding paper of diffusion models
  - Textual Inversion: https://arxiv.org/abs/2208.01618 (impl: https://github.com/rinongal/textual_inversion)
    -  Stable Conceptualizer https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb
  - 2017: Attention is all you need
  - https://dreambooth.github.io/
    - productized as dreambooth https://twitter.com/psuraj28/status/1575123562435956740
    - https://github.com/JoePenna/Dreambooth-Stable-Diffusion ([examples](https://twitter.com/rainisto/status/1584881850933456898))
    - from huggingface diffusers https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb
    - https://twitter.com/rainisto/status/1584881850933456898
      - Commercial offerings
        - https://avatarai.me/
        - https://www.astria.ai/ (formerly https://www.strmr.com/)
        - https://twitter.com/rohanarora_/status/1580413809516511232?s=20&t=XxjfadtkVM8TOvg5EYFCrw
  - [very good BLOOM model overview](https://www.youtube.com/watch?v=3EjtHs_lXnk)

## Products

- Lexica (search + gen)
- Pixelvibe (search + gen) https://twitter.com/lishali88/status/1595029444988649472

product placement
- Pebbley -  inpainting  https://twitter.com/alfred_lua/status/1610641101265981440?s=46&t=RMPT1jJedELVkL2Aby-40g
- Flair AI https://twitter.com/mickeyxfriedman/status/1613251965634465792
- scale AI forge https://twitter.com/alexandr_wang/status/1614998087176720386

## Stable Diffusion prompts

The basic intuition of Stable Diffusion is that you have to add descriptors to get what you want. 

From [here](https://news.ycombinator.com/item?id=33086085):

<details>
	<summary>
		"George Washington riding a Unicorn in Times Square"
	</summary>

  ![image](https://user-images.githubusercontent.com/6764957/194002068-bf0345a6-1826-4a41-8c39-47fee653e207.png)

</details>

<details>
	<summary>
		George Washington riding a unicorn in Times Square, cinematic composition, concept art, digital illustration, detailed
	</summary>


  ![image](https://user-images.githubusercontent.com/6764957/194002170-748bfe81-8e60-4b32-8a43-162f470b9d9f.png)


</details>


Prompts might go in the form of 

```
[Prefix] [Subject], [Enhancers]
```

Adding the right enhancers can really tweak the outcome:

![image](https://user-images.githubusercontent.com/6764957/188303877-4555e026-4da5-4f22-b7f5-2972425350ba.png)

### SD v2 prompts

SD2 Prompt Book from Stability: https://stability.ai/sdv2-prompt-book

### SD 1.4 vs 1.5 comparisons

- https://twitter.com/TomLikesRobots/status/1583836870445670401
- https://twitter.com/multimodalart/status/1583404683204648960

### Distilled Stable Diffusion

- https://twitter.com/EMostaque/status/1598131202044866560 20x speed up, convergence in 1-4 steps
	- https://arxiv.org/abs/2210.03142
	- "We already reduced time to gen 50 steps from 5.6s to 0.9s working with nvidia"
	- https://arxiv.org/abs/2210.03142
		- For diffusion models trained on the latent-space (e.g., Stable Diffusion), our approach is able to generate high-fidelity images using as few as 1 to 4 denoising steps, accelerating inference by at least 10-fold compared to existing methods on ImageNet 256x256 and LAION datasets. We further demonstrate the effectiveness of our approach on text-guided image editing and inpainting, where our distilled model is able to generate high-quality results using as few as 2-4 denoising steps.
- Stable diffusion speed progress  https://www.listennotes.com/podcasts/the-logan-bartlett/ep-46-stability-ai-ceo-emad-8PQIYcR3r2i/
	- Aug 2022 - 5.6s/image
	- Dec 2022 - 0.9s/image
	- Jan 2022 - 30 images/s (100x speed increase)

## SD2 vs SD1 user notes


- Comparisons
  - https://twitter.com/dannypostmaa/status/1595612366770954242?s=46
  - https://www.reddit.com/r/StableDiffusion/comments/z3ferx/xy_plot_comparisons_of_sd_v15_ema_vs_sd_20_x768/
  - compare it yourself https://app.gooey.ai/CompareText2Img/?example_id=1uONp1IBt0Y
  - depth2img produces more coherence for animations https://www.reddit.com/r/StableDiffusion/comments/zk32dg/a_quick_demo_to_show_how_structurally_coherent/
- https://twitter.com/EMostaque/status/1595731398450634755
  - V2 prompts different and will take a while for folk to get used to. V2 is trained on two models, a generator model and a image-to-text model (CLIP).
  - We supported @laion_ai in their creation of an OpenCLIP Vit-H14 https://twitter.com/wightmanr/status/1570503598538379264
  - We released two variants of the 512 model which I would recommend folk dig into, especially the -v model.. More on these soon. The 768 model I think will improve further from here as the first of its type, we will have far more regular updates, releases and variants from here
  - Elsewhere I would highly recommend folk dig into the depth2img model, fun things coming there. 3D maps will improve, particularly as we go onto 3D models and some other fun stuff to be announced in the new year. These models are best not zero-shot, but as part of a process  
- Stable Diffusion 2.X was trained on LAION-5B as opposed to "laion-improved-aesthetics" (a subset of laion2B-en). for Stable Diffusion 1.X.



## Hardware requirements

- https://news.ycombinator.com/item?id=32642255#32646761
  - For something like this, you ideally would want a powerful GPU with 12-24gb VRAM. 
  - A $500 RTX 3070 with 8GB of VRAM can generate 512x512 images with 50 steps in 7 seconds.
- https://huggingface.co/blog/stable_diffusion_jax uper fast inference on Google TPUs, such as those available in Colab, Kaggle or Google Cloud Platform - 8 images in 8 seconds
- Intel CPUs: https://github.com/bes-dev/stable_diffusion.openvino
- aws ec2 guide https://aws.amazon.com/blogs/architecture/an-elastic-deployment-of-stable-diffusion-with-discord-on-aws/

## Stable Diffusion

stable diffusion specific notes

Required reading:
- param intuition https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
- CLI commands https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/#script-options




### SD Distros

- **Installer Distros**: Programs that bundle Stable Diffusion in an installable program, no separate setup and the least amount of git/technical skill needed, usually bundling one or more UI
  - iPad: [Draw Things App](https://apps.apple.com/app/id6444050820)
  - [Diffusion Bee](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui) (open source): Diffusion Bee is the easiest way to run Stable Diffusion locally on your M1 Mac. Comes with a one-click installer. No dependencies or technical knowledge needed.
  - https://github.com/cmdr2/stable-diffusion-ui: Easiest 1-click way to install and use Stable Diffusion on your own computer. Provides a browser UI for generating images from text prompts and images. Just enter your text prompt, and see the generated image. (Linux, Windows, no Mac). 
  - https://nmkd.itch.io/t2i-gui: A basic (for now) Windows 10/11 64-bit GUI to run Stable Diffusion, a machine learning toolkit to generate images from text, locally on your own hardware. As of right now, this program only works on Nvidia GPUs! AMD GPUs are not supported. In the future this might change. 
  - [imaginAIry 🤖🧠](https://github.com/brycedrennan/imaginAIry) (SUPPORTS SD 2.0!): Pythonic generation of stable diffusion images with just `pip install imaginairy`. "just works" on Linux and macOS(M1) (and maybe windows). Memory efficiency improvements, prompt-based editing, face enhancement, upscaling, tiled images, img2img, prompt matrices, prompt variables, BLIP image captions, comes with dockerfile/colab.  Has unit tests.
    - Note: it goes a lot faster if you run it all inside the included aimg CLI, since then it doesn't have to reload the model from disk every time
  - [Fictiverse/Windows-GUI](https://github.com/Fictiverse/StableDiffusion-Windows-GUI): A windows interface for stable diffusion
  - SD from Apple Core ML https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon https://github.com/apple/ml-stable-diffusion
	  - [Gauss macOS native app](https://github.com/justjake/Gauss) (open source)
	  - https://sindresorhus.com/amazing-ai SindreSorhus exclusive for M1/M2
  - https://www.charl-e.com/ (open source): Stable Diffusion on your Mac in 1 click. ([tweet](https://twitter.com/charliebholtz/status/1571138577744138240))
  - https://github.com/razzorblade/stable-diffusion-gui: dormant now.
- **Web Distros**
  - [web stable diffusion](https://github.com/mlc-ai/web-stable-diffusion) - running in browser
  - Gooey - https://app.gooey.ai/CompareText2Img/?example_id=1uONp1IBt0Y
  - https://playgroundai.com/create UI for DallE and Stable Diffusion
  - https://www.phantasmagoria.me/
  - https://www.mage.space/
  - https://inpainter.vercel.app 
  - https://dreamlike.art/ has img2img
  - https://inpainter.vercel.app/paint for inpainting
  - https://promptart.labml.ai/feed
  - https://www.strmr.com/ dreambooth tuning for $3
  - https://www.findanything.app browser extension that adds SD predictions alongside Google search
  - https://www.drawanything.app 
  - https://huggingface.co/spaces/huggingface-projects/diffuse-the-rest draw a thing, diffuse the rest! 
  - https://creator.nolibox.com/guest open source https://github.com/carefree0910/carefree-creator
	-  An **infinite draw board** for you to save, review and edit all your creations.
	- Almost EVERY feature about Stable Diffusion (txt2img, img2img, sketch2img, **variations**, outpainting, circular/tiling textures, sharing, ...).
	- Many useful image editing methods (**super resolution**, inpainting, ...).
	- Integrations of different Stable Diffusion versions (waifu diffusion, ...).
	- GPU RAM optimizations, which makes it possible to enjoy these features with an NVIDIA GeForce GTX 1080 Ti
  - https://replicate.com/stability-ai/stable-diffusion Predictions run on Nvidia A100 GPU hardware. Predictions typically complete within 5 seconds.
  - https://replicate.com/cjwbw/stable-diffusion-v2
  - https://deepinfra.com/
- **iPhone/iPad Distros**
  - https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820
  - another attempt that was paused https://www.cephalopod.studio/blog/on-creating-an-on-device-stable-diffusion-app-amp-deciding-not-to-release-it-adventures-in-ai-ethics
- **Finetuned Distros**
  - [Arcane Diffusion](https://huggingface.co/spaces/anzorq/arcane-diffusion) a fine-tuned Stable Diffusion model trained on images from the TV Show Arcane.
  - [Spider-verse Diffusion](https://huggingface.co/nitrosocke/spider-verse-diffusion) rained on movie stills from Sony's Into the Spider-Verse. Use the tokens spiderverse style in your prompts for the effect.
  - [Simpsons Dreambooth](https://www.reddit.com/r/StableDiffusion/comments/zghkj0/new_dreambooth_model_the_simpsons/)
  - https://huggingface.co/ItsJayQz 
	  - Roy PopArt Diffusion 2 🐢
	  - GTA5 Artwork Diffusion 😻
	  - Firewatch Diffusion 1 💻 
	  - Civilizations 6 Diffusion 1 🔥 
	  - Classic Telltale Diffusion 3 😻 
	  - Marvel WhatIf Diffusion
  - [Texture inpainting](https://twitter.com/StableDiffusion/status/1580840640501649408)
  - How to finetune your own
    - Naruto version https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-naruto-character-edition
    - Pokemon https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda
    - https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-textual-inversion-b995d7ecc095
- **Twitter Bots**
  - https://twitter.com/diffusionbot
  - https://twitter.com/m1guelpf/status/1569487042345861121
- **Windows "retard guides"**
  - https://rentry.org/voldy
  - https://rentry.org/GUItard

### SD Major forks

Main Stable Diffusion repo: https://github.com/CompVis/stable-diffusion
- Tensorflow/Keras impl: https://github.com/divamgupta/stable-diffusion-tensorflow
- Diffusers library: https://github.com/huggingface/diffusers ([Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb))

| Name/Link 	| Stars 	| Description 	|
|---	|---	|---	|
| [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 	| 26000 	| The most well known fork. features: https://github.com/AUTOMATIC1111/stable-diffusion-webui#features launch announcement https://www.reddit.com/r/StableDiffusion/comments/x28a76/stable_diffusion_web_ui/. M1 mac instructions https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon 	|
| [Disco Diffusion](https://github.com/alembics/disco-diffusion) 	| 6400 	| A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations. 	|
| [sd-webui](https://github.com/sd-webui/stable-diffusion-webui) (formerly hlky fork) 	| 6000 	| A fully-integrated and easy way to work with Stable Diffusion right from a browser window. Long list of UI and SD features (incl textual inversion, alternative samplers, prompt matrix): https://github.com/sd-webui/stable-diffusion-webui#project-features 	|
| [InvokeAI](https://github.com/invoke-ai/InvokeAI) (formerly lstein fork) 	| 8800 	| This version of Stable Diffusion features a slick WebGUI, an interactive command-line script that combines text2img and img2img functionality in a "dream bot" style interface, and multiple features and other enhancements. It runs on Windows, Mac and Linux machines, with GPU cards with as little as 4 GB of RAM. Universal Canvas (see [youtube](https://www.youtube.com/watch?v=hIYBfDtKaus&lc=UgydbodXO5Y9w4mnQHN4AaABAg.9j4ORX-gv-w9j78Muvp--w)) 	|
| [XavierXiao/Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) 	| 4900 	| Implementation of Dreambooth (https://arxiv.org/abs/2208.12242) with Stable Diffusion. Dockerized: https://github.com/smy20011/dreambooth-docker 	|
| [Basujindal: Optimized Stable Diffusion](https://github.com/basujindal/stable-diffusion) 	| 2600 	| This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed. img2img and txt2img and inpainting under 2.4GB VRAM  	|
| [stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity) 	| 2800 	| Outpainting with Stable Diffusion on an infinite canvas. This project mainly works as a proof of concept.  	|
| [Waifu Diffusion](https://github.com/harubaru/waifu-diffusion) ([huggingface](https://huggingface.co/hakurei/waifu-diffusion), [replicate](https://replicate.com/cjwbw/waifu-diffusion)) 	| 1600 	| stable diffusion finetuned on weeb stuff. "A model trained on danbooru (anime/manga drawing site with also lewds and nsfw on it) over 56k images.Produces FAR BETTER results if you're interested in getting manga and anime stuff out of stable diffusion." 	|
| [AbdBarho/stable-diffusion-webui-docker](https://github.com/AbdBarho/stable-diffusion-webui-docker) 	| 1600 	| Easy Docker setup for Stable Diffusion with both Automatic1111 and hlky UI included. HOWEVER - no mac support yet https://github.com/AbdBarho/stable-diffusion-webui-docker/issues/35 	|
| [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) 	| 3200 	|  +25-50% speed increase + memory efficient + DreamBooth 	|
| [nolibox/carefree-creator](https://github.com/carefree0910/carefree-creator) 	| 1800 	|  An infinite draw board for you to save, review and edit all your creations. Almost EVERY feature about Stable Diffusion (txt2img, img2img, sketch2img, variations, outpainting, circular/tiling textures, sharing, ...). Many useful image editing methods (super resolution, inpainting, ...). Integrations of different Stable Diffusion versions (waifu diffusion, ...). GPU RAM optimizations, which makes it possible to enjoy these features with an NVIDIA GeForce GTX 1080 Ti! It might be fair to consider this as: An AI-powered, open source Figma. A more 'interactable' Hugging Face Space. A place where you can try all the exciting and cutting-edge models, together. 	|
| [imaginAIry 🤖🧠](https://github.com/brycedrennan/imaginAIry) | 1600 | Pythonic generation of stable diffusion images with just `pip install imaginairy`. "just works" on Linux and macOS(M1) (and maybe windows). Memory efficiency improvements, prompt-based editing, face enhancement, upscaling, tiled images, img2img, prompt matrices, prompt variables, BLIP image captions, comes with dockerfile/colab.  Has unit tests. |
| [neonsecret/stable-diffusion](https://github.com/neonsecret/stable-diffusion) 	| 582 	| This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed. Also I invented the sliced atttention technique, which allows to push the model's abilities even further. It works by automatically determining the slice size from your vram and image size and then allocating it one by one accordingly. You can practically generate any image size, it just depends on the generation speed you are willing to sacrifice. 	|
| [Deforum Stable Diffusion](https://github.com/deforum/stable-diffusion) 	| 591 	| Animating prompts with stable diffusion.  Weighted Prompts,  Perspective 2D Flipping, Dynamic Video Masking, Custom MATH expressions, Waifu and Robo Diffusion Models. [twitter, changelog](https://twitter.com/deforum_art/status/1576330236194525184?s=20&t=36133FXROv0CMGHOoSxHyg). replicate demo: https://replicate.com/deforum/deforum_stable_diffusion 	|
| [Maple Diffusion](https://github.com/madebyollin/maple-diffusion) 	| 550 	| Maple Diffusion runs Stable Diffusion models locally on macOS / iOS devices, in Swift, using the MPSGraph framework (not Python). [Matt Waller working on CoreML impl](https://twitter.com/divamgupta/status/1583482195192459264) 	|
| [Doggettx/stable-diffusion](https://github.com/Doggettx/stable-diffusion) 	| 158 	| Allows to use resolutions that require up to 64x more VRAM than possible on the default CompVis build. 	|
| [Doohickey Diffusion](https://twitter.com/StableDiffusion/status/1580840624206798848) 	| 29 	| CLIP guidance, perceptual guidance, Perlin initial noise, and other features.  	|

https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py with Vid2Vid

Future Diffusion  https://huggingface.co/nitrosocke/Future-Diffusion https://twitter.com/Nitrosocke/status/1599789199766716418

#### SD in Other languages

- Chinese: https://twitter.com/_akhaliq/status/1572580845785083906
- Japanese: https://twitter.com/_akhaliq/status/1571977273489739781
  - https://huggingface.co/blog/japanese-stable-diffusion
- DALL-E's inherent multilingualness https://twitter.com/Merzmensch/status/1551179292704399360 (we dont know the CLIP Vit-H embeddings details)
  
#### Other Lists of Forks

- https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/
- https://www.reddit.com/r/StableDiffusion/comments/xcclmf/comment/io6u03s/?utm_source=reddit&utm_medium=web2x&context=3
- https://techgaun.github.io/active-forks/index.html#CompVis/stable-diffusion

SD Model search and ratings: https://civitai.com/

Dormant projects, for historical/research interest:

- https://colab.research.google.com/drive/1AfAmwLMd_Vx33O9IwY2TmO9wKZ8ABRRa		
- https://colab.research.google.com/drive/1kw3egmSn-KgWsikYvOMjJkVDsPLjEMzl		
- [bfirsh/stable-diffusion](https://github.com/bfirsh/stable-diffusion)	No longer actively maintained byt was the first to work on M1 Macs - [blog](https://replicate.com/blog/run-stable-diffusion-on-m1-mac), [tweet](https://twitter.com/levelsio/status/1565731907664478209), can also look at `environment-mac.yaml` from https://github.com/fragmede/stable-diffusion/blob/mps_consistent_seed/environment-mac.yaml

#### Misc SD UI's

UI's that dont come with their own SD distro, just shelling out to one

| UI Name/Link 	| Stars 	| Self-Description 	|
|---	|---	|---	|
| [ahrm/UnstableFusion](https://github.com/ahrm/UnstableFusion) 	| 815 	| UnstableFusion is a desktop frontend for Stable Diffusion which combines image generation, inpainting, img2img and other image editing operation into a seamless workflow.  https://www.youtube.com/watch?v=XLOhizAnSfQ&t=1s 	|
| [stable-diffusion-2-gui](https://github.com/qunash/stable-diffusion-2-gui/) 	| 262 	| Lightweight Stable Diffusion v 2.1 web UI: txt2img, img2img, depth2img, inpaint and upscale4x. 	|
| [breadthe/sd-buddy](https://github.com/breadthe/sd-buddy/) 	| 165 	| Companion desktop app for the self-hosted M1 Mac version of Stable Diffusion, with Svelte and Tauri 	|
| [leszekhanusz/diffusion-ui](https://github.com/leszekhanusz/diffusion-ui) 	| 65 	| This is a web interface frontend for the generation of images using diffusion models.<br><br>The goal is to provide an interface to online and offline backends doing image generation and inpainting like Stable Diffusion. 	|
| [GenerationQ](https://github.com/westoncb/generation-q) 	| 21 	| GenerationQ (for "image generation queue") is a cross-platform desktop application (screens below) designed to provide a general purpose GUI for generating images via text2img and img2img models. Its primary target is Stable Diffusion but since there is such a variety of forked programs with their own particularities, the UI for configuring image generation tasks is designed to be generic enough to accommodate just about any script (even non-SD models). 	|


### SD Prompt galleries and search engines

- 🌟 [Lexica](https://lexica.art/): Content-based search powered by OpenAI's CLIP model. **Seed**, CFG, Dimensions.
- [PromptFlow](https://promptflow.co): Search engine that allows for on-demand generation of new results. Search 10M+ of AI art and prompts generated by DALL·E 2, Midjourney, Stable Diffusion
- https://synesthetic.ai/ SD focused
- https://visualise.ai/ Create and share image prompts. DALL-E, Midjourney, Stable Diffusion
- https://nyx.gallery/
- [OpenArt](https://openart.ai/discovery?dataSource=sd): Content-based search powered by OpenAI's CLIP model. Favorites.
- [PromptHero](https://prompthero.com/): [Random wall](https://prompthero.com/random). **Seed**, CFG, Dimensions, Steps. Favorites.
- [Libraire](https://libraire.ai/): **Seed**, CFG, Dimensions, Steps.
- [Krea](https://www.krea.ai/): modifiers focused UI. Favorites. Gives prompt suggestions and allows to create prompts over Stable diffusion, Waifu Diffusion and Disco diffusion. Really quick and useful
	- browse https://atlas.nomic.ai/map/809ef16a-5b2d-4291-b772-a913f4c8ee61/9ed7d171-650b-4526-85bf-3592ee51ea31
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
- Prompt marketplace: [Prompt Hunt](https://www.prompthunt.com/)
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
	- https://github.com/kyrick/cog-prompt-parrot
- [cmdr2](https://github.com/cmdr2/stable-diffusion-ui): 1-click SD installation with image modifiers selection.

### Img2prompt - Reverse Prompt Engineering

- [img2prompt](https://replicate.com/methexis-inc/img2prompt) Replicate by [methexis-inc](https://replicate.com/methexis-inc): Optimized for SD (clip ViT-L/14).
- [CLIP Interrogator](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) by [@pharmapsychotic](https://twitter.com/pharmapsychotic): select ViTL14 CLIP model.
  - https://huggingface.co/spaces/pharma/sd-prism Sends an image in to CLIP Interrogator to generate a text prompt which is then run through Stable Diffusion to generate new forms of the original!
- CLIPSeg -> image segmentation
- [CLIP Artist Evaluator colab](https://colab.research.google.com/github/lowfuel/CLIP_artists/blob/main/CLIP_Evaluator.ipynb)
- [BLIP](https://huggingface.co/spaces/Salesforce/BLIP)

### Explore Artists, styles, and modifiers

See https://github.com/sw-yx/prompt-eng/blob/main/PROMPTS.md for more details and notes

- [Artist Style Studies](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d) & [Modifier Studies](https://www.notion.so/2b07d3195d5948c6a7e5836f9d535592) by [parrot zone](https://www.notion.so/74a5c04d4feb4f12b52a41fc8750b205): **[Gallery](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d)**, [Style](https://www.notion.so/e28a4f8d97724f14a784a538b8589e7d), [Spreadsheet](https://docs.google.com/spreadsheets/d/14xTqtuV3BuKDNhLotB_d1aFlBGnDJOY0BRXJ8-86GpA/edit#gid=0)
- [Clip retrieval](https://knn5.laion.ai/): search [laion-5b](https://laion.ai/blog/laion-5b/) dataset.
- [Datasette](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls): [image search](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/images); image-count sort by [artist](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/artists?_sort_desc=image_counts), [celebrities](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/celebrities?_sort_desc=image_counts), [characters](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/characters?_sort_desc=image_counts), [domain](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/domain?_sort_desc=image_counts)
- [Visual](https://en.wikipedia.org/wiki/Category:Visual_arts) [arts](https://en.wikipedia.org/wiki/Category:The_arts): [media](https://en.wikipedia.org/wiki/Category:Visual_arts_media) [list](https://en.wikipedia.org/wiki/List_of_art_media), [related](https://en.wikipedia.org/wiki/Category:Arts-related_lists); [Artists](https://en.wikipedia.org/wiki/Category:Artists) [list](https://en.wikipedia.org/wiki/Category:Lists_of_artists) by [genre](https://en.wikipedia.org/wiki/Category:Artists_by_genre), [medium](https://en.wikipedia.org/wiki/Category:Artists_by_medium); [Portal](https://en.wikipedia.org/wiki/Portal:The_arts)

### SD Prompt Tools directories and guides

- https://diffusiondb.com/ 543 Stable Diffusion systems
- Useful Prompt Engineering tools and resources https://np.reddit.com/r/StableDiffusion/comments/xcrm4d/useful_prompt_engineering_tools_and_resources/
- [Tools and Resources for AI Art](https://pharmapsychotic.com/tools.html) by [pharmapsychotic](https://www.reddit.com/user/pharmapsychosis)
- [Akashic Records](https://github.com/Maks-s/sd-akashic#prompts-toc)
- [Awesome Stable-Diffusion](https://github.com/Maks-s/sd-akashic#prompts-toc)
- Install Stable Diffusion 2.1 purely through the terminal https://medium.com/@diogo.ribeiro.ferreira/how-to-install-stable-diffusion-2-0-on-your-pc-f92b9051b367

### Finetuning/Dreambooth

How to finetune
- https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda


Stable Diffusion + Midjourney
- https://www.reddit.com/r/StableDiffusion/comments/z622mp/comment/ixyy2qz/?utm_source=share&utm_medium=web2x&context=3

Embeddings/Textual Inversion
- knollingcase https://huggingface.co/ProGamerGov/knollingcase-embeddings-sd-v2-0
- https://www.reddit.com/r/StableDiffusion/comments/zxkukk/detailed_guide_on_training_embeddings_on_a/
	- -   A model is a 2GB+ file that can do basically anything. It takes a lot of VRAM to train and has a large file size.
	-   A hypernetwork is an 80MB+ file that sits on top of a model and can learn new things not present in the base model. It is relatively easy to train, but is typically less flexible than an embedding when using it in other models.	    
	-   An embedding is a 4KB+ file (yes, 4 kilobytes, it's very small) that can be applied to any model that uses the same base model, which is typically the base stable diffusion model. It cannot learn new content, rather it creates magical keywords behind the scenes that tricks the model into creating what you want.
- "hyper models" 
	- https://twitter.com/zhansheng/status/1595456793068568581?s=46&t=Nd874xTjwniEuGu2d1toQQ
	- Introducing HyperTuning: Using a hypermodel to generate parameters for frozen downstream models. This allows us to adapt models to new tasks *without* back-prop! Paper: arxiv.org/abs/2211.12485
- textual inversion https://www.reddit.com/r/StableDiffusion/comments/zpcutz/breakdown_of_how_i_make_embeddings_for_my/
- hypernetworks https://www.reddit.com/r/StableDiffusion/comments/zntxoz/invisible_hypernetwork/

Dreambooth
- https://bytexd.com/how-to-use-dreambooth-to-fine-tune-stable-diffusion-colab/
- https://replicate.com/blog/dreambooth-api
- https://huggingface.co/spaces/multimodalart/dreambooth-training (tech notes https://twitter.com/multimodalart/status/1598260506460311557)
- https://github.com/ShivamShrirao/diffusers
	- produces https://twitter.com/rainisto/status/1600563803912929280
- Art project - faking entire instagram profile for a month using dreambooth https://www.reddit.com/r/StableDiffusion/comments/zkvnyx/using_stablediffusion_and_dreambooth_i_faked_my/

Trained examples
- Pixel art animation spritesheets
	- https://ps.reddit.com/r/StableDiffusion/comments/yj1kbi/ive_trained_a_new_model_to_output_pixel_art/
	- https://twitter.com/kylebrussell/status/1587477169474789378
- [Dreambooth 2D 3D icons](https://www.reddit.com/r/StableDiffusion/comments/zmomeu/creating_a_stable_diffusion_dreambooth_2d_to_3d/) (https://pixelpoint.io/blog/ms-fluent-emoji-style-fine-tune-on-stable-diffusion/)
- Analog Diffusion https://www.reddit.com/r/StableDiffusion/comments/zi3g5x/new_15_dreambooth_model_analog_diffusion_link_in/ and [more exampels](https://www.reddit.com/r/StableDiffusion/comments/zkqtqb/im_in_love_with_the_analog_diffusion_10_model/)
	- This is a dreambooth model trained on a diverse set of analog photographs.
	- comparison with other photoreal models https://www.reddit.com/r/StableDiffusion/comments/102ljfh/comment/j2tuw2p/?utm_source=reddit&utm_medium=web2x&context=3
		- dreamlike photoreal https://www.reddit.com/r/StableDiffusion/comments/102t0av/new_photorealistic_model_dreamlike_photoreal_20/
- Protogen
	- https://civitai.com/models/3627/protogen-v22-official-release
	- https://www.reddit.com/r/StableDiffusion/comments/1003bsv/protogen_v22_official_release/
	- https://www.reddit.com/r/StableDiffusion/comments/100fmx6/protogen_x34_official_release/

### ControlNet

- https://huggingface.co/spaces/hysts/ControlNet
- inspirations
	- https://www.reddit.com/r/StableDiffusion/comments/11ku886/controlnet_unlimited_album_covers_graphic_design/
	- https://www.reddit.com/r/StableDiffusion/comments/11bp30o/tech_companies_as_charcuterie_boards_controlnet/

#### SD Tooling

- AI Dreamer iOS/macOS app https://apps.apple.com/us/app/ai-dreamer/id1608856807
- SD's DreamStudio https://beta.dreamstudio.ai/dream
- Stable Worlds: [colab](https://colab.research.google.com/drive/1RXRrkKUnpNiPCxTJg0Imq7sIM8ltYFz2?usp=sharing) for 3d stitched worlds via StableDiffusion https://twitter.com/NaxAlpha/status/1578685845099290624
- Hardmaru Highres Inpainting experiment
	- https://twitter.com/hardmaru/status/1608008214875967489?s=20
	- https://github.com/hardmaru/image-notebook/tree/main/stable-diffusion-2
- Midjourney + SD: https://twitter.com/EMostaque/status/1561917541743841280
- [Nightcafe Studio](https://creator.nightcafe.studio/stable-diffusion-image-generator)
- misc
  - words -> mask -> replacement. utomatic mask generation with CLIPSeg https://twitter.com/NielsRogge/status/1593645630412402688


## How SD Works - Internals and Studies

- How SD works
  - SD quickstart https://www.reddit.com/r/StableDiffusion/comments/xvhavo/made_an_easy_quickstart_guide_for_stable_diffusion/   
  - https://huggingface.co/blog/stable_diffusion
  - https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
    - tinygrad impl https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
    - Diffusion with offset noise https://www.crosslabs.org//blog/diffusion-with-offset-noise
  - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
  - FastAI course https://www.fast.ai/posts/part2-2022-preview.html
  - https://twitter.com/johnowhitaker/status/1565710033463156739
  - https://twitter.com/ai__pub/status/1561362542487695360
  - https://twitter.com/JayAlammar/status/1572297768693006337
    - https://jalammar.github.io/illustrated-stable-diffusion/
  - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
  - annotated SD implementation https://twitter.com/labmlai/status/1571080112459878401
	  - https://nn.labml.ai/diffusion/stable_diffusion/scripts/text_to_image.html
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

## InstructPix2Pix

- https://www.timothybrooks.com/instruct-pix2pix
- Pix2Pixzero - https://pix2pixzero.github.io/
	- We propose pix2pix-zero, a diffusion-based image-to-image approach that allows users to specify the edit direction on-the-fly (e.g., cat to dog). Our method can directly use pre-trained text-to-image diffusion models, such as Stable Diffusion, for editing real and synthetic images while preserving the input image's structure. Our method is training-free and prompt-free, as it requires neither manual text prompting for each input image nor costly fine-tuning for each task.


## Extremely detailed prompt examples

- [dark skinned Johnny Storm young male superhero of the fantastic four, full body, flaming dreadlock hair, blue uniform with the number 4 on the chest in a round logo, cinematic, high detail, no imperfections, extreme realism, high detail, extremely symmetric facial features, no distortion, clean, also evil villians fighting in the background, by Stan Lee](https://lexica.art/prompt/d622e029-176d-42b7-a437-39ccf1952b71)
- [(extremely detailed CG unity 8k wallpaper), full shot body photo of a (((beautiful badass woman soldier))) with ((white hair)), ((wearing an advanced futuristic fight suit)), ((standing on a battlefield)), scorched trees and plants in background, sexy, professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, by midjourney and greg rutkowski, realism, beautiful and detailed lighting, shadows, by Jeremy Lipking, by Antonio J. Manzanedo, by Frederic Remington, by HW Hansen, by Charles Marion Russell, by William Herbert Dunton](https://www.reddit.com/r/StableDiffusion/comments/100tp0v/protogenx34_has_absolutely_amazing_detail/)
- [dark and gloomy full body 8k unity render, female teen cyborg, Blue yonder hair, wearing broken battle armor, at cluttered and messy shack , action shot, tattered torn shirt, porcelain cracked skin, skin pores, detailed intricate iris, very dark lighting, heavy shadows, detailed, detailed face, (vibrant, photo realistic, realistic, dramatic, dark, sharp focus, 8k)](https://www.reddit.com/r/StableDiffusion/comments/102nn3s/closest_i_can_get_to_midjourney_style_no_artists/)

### Solving Hands

- Negative prompts: ugly, disfigured, too many fingers, too many arms, too many legs, too many hands

## Midjourney prompts

- https://twitter.com/textfiles/status/1591583867835645958?s=20&t=NPVEYUcYgumQS9KNKwtuuQ

## Misc

- Craiyon/Dall-E Mini
  - https://github.com/borisdayma/dalle-mini
  - https://news.ycombinator.com/item?id=33668023a
  - GitHub: https://github.com/borisdayma/dalle-mini
  - Hugging Face Demo: https://huggingface.co/spaces/flax-community/dalle-mini
  - NYT article: https://www.nytimes.com/2022/04/06/technology/openai-images-dall-e.html
- Structured Diffusion https://twitter.com/WilliamWangNLP/status/1602722552312262656
	- great examples better than StableDiffusion
- Imagen
  - https://www.assemblyai.com/blog/how-imagen-actually-works
  - https://www.youtube.com/watch?v=R_f-v6prMqI
- Nvidia eDiffi (unreleased)
	- https://deepimagination.cc/eDiff-I/
	- https://twitter.com/search?q=https%3A%2F%2Ftwitter.com%2F_akhaliq%2Fstatus%2F1587971650007564289&src=typed_query
- Artist protests
	- https://vmst.io/@selzero/109512557990367884