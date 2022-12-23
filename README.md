# AI Notes

notes on AI state of the art, with a focus on generative and large language models. These are the "raw materials" for the https://lspace.swyx.io/ newsletter.

> This repo used to be called https://github.com/sw-yx/prompt-eng, but was renamed because [Prompt Engineering is Overhyped](https://twitter.com/swyx/status/1596184757682941953).

This Readme is just the high level overview of the space; you should see the most updates in the OTHER markdown files in this repo:

- `IMAGE_GEN.md` - the most developed file, with the heaviest emphasis notes on Stable Diffusion, and some on midjourney and dalle.
- `TEXT.md` - text generation, mostly with GPT3
- `CODE.md` - codegen models, like Copilot
- stubs - very small/lightweight proto pages
  - `AGENTS.md` - tracking "agentic AI"
  - `AUDIO.md` - tracking audio (transcription + generation)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [Motivational Use Cases](#motivational-use-cases)
- [Top AI Reads](#top-ai-reads)
- [Communities](#communities)
- [People](#people)
- [Misc](#misc)
- [Quotes, Reality & Demotivation](#quotes-reality--demotivation)
- [Infrastructure](#infrastructure)
- [Legal, Ethics, and Privacy](#legal-ethics-and-privacy)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Motivational Use Cases

- images
  - https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts/
  - [3D MRI synthetic brain images](https://twitter.com/Warvito/status/1570691960792580096?) - [positive reception from neuroimaging statistician](https://twitter.com/danCMDstat/status/1572312699853312000?s=20&t=x-ouUbWA5n0-PxTGZcy2iA)
  - [multiplayer stable diffusion](https://huggingface.co/spaces/huggingface-projects/stable-diffusion-multiplayer?roomid=room-0)
- video
  - img2img of famous movie scenes ([lalaland](https://twitter.com/TomLikesRobots/status/1565678995986911236))
    - [img2img transforming actor](https://twitter.com/LighthiserScott/status/1567355079228887041?s=20&t=cBH4EGPC4r0Earm-mDbOKA) with ebsynth + koe_recast
  - virtual fashion ([karenxcheng](https://twitter.com/karenxcheng/status/1564626773001719813))
  - [seamless tiling images](https://twitter.com/replicatehq/status/1568288903177859072?s=20&t=sRd3HRehPMcj1QfcOwDMKg)
  - evolution of scenes ([xander](https://twitter.com/xsteenbrugge/status/1558508866463219712))
  - outpainting https://twitter.com/orbamsterdam/status/1568200010747068417?s=21&t=rliacnWOIjJMiS37s8qCCw
  - webUI img2img collaboration https://twitter.com/_akhaliq/status/1563582621757898752
  - image to video with rotation https://twitter.com/TomLikesRobots/status/1571096804539912192
  - "prompt paint" https://twitter.com/1littlecoder/status/1572573152974372864
  - audio2video animation of your face https://twitter.com/siavashg/status/1597588865665363969
  - physical toys to 3d model + animation https://twitter.com/sergeyglkn/status/1587430510988611584
  - music videos 
    - [video killed the radio star](https://www.youtube.com/watch?v=WJaxFbdjm8c), [colab](https://colab.research.google.com/github/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb) This uses OpenAI's Whisper speech-to-text, allowing you to take a YouTube video & create a Stable Diffusion animation prompted by the lyrics in the YouTube video
    - [Stable Diffusion Videos](https://colab.research.google.com/github/nateraw/stable-diffusion-videos/blob/main/stable_diffusion_videos.ipynb) generates videos by interpolating between prompts and audio
  - direct text2video project
    - https://twitter.com/_akhaliq/status/1575546841533497344
    - https://makeavideo.studio/ - explorer https://webvid.datasette.io/webvid/videos
    - https://phenaki.video/
    - https://github.com/THUDM/CogVideo
    - https://imagen.research.google/video/
- text-to-3d https://twitter.com/_akhaliq/status/1575541930905243652
  -  https://dreamfusion3d.github.io/
  -  open source impl: https://github.com/ashawkey/stable-dreamfusion
    - demo https://twitter.com/_akhaliq/status/1578035919403503616
-  text products
  - Jasper
  - gpt3 email https://github.com/sw-yx/gpt3-email
  - gpt3() in google sheet [2020](https://twitter.com/pavtalk/status/1285410751092416513?s=20&t=ppZhNO_OuQmXkjHQ7dl4wg), [2022](https://twitter.com/shubroski/status/1587136794797244417) - [sheet](https://docs.google.com/spreadsheets/d/1YzeQLG_JVqHKz5z4QE9wUsYbLoVZZxbGDnj7wCf_0QQ/edit)
  - https://www.summari.com/ Summari helps busy people read more
- sequoia market map https://twitter.com/sonyatweetybird/status/1584580362339962880
- base10 market map https://twitter.com/letsenhance_io/status/1594826383305449491
  - game assets - emad thread https://twitter.com/EMostaque/status/1591436813750906882

## Top AI Reads

The more advanced GPT3 reads have been split out to https://github.com/sw-yx/prompt-eng/blob/main/GPT.md

- https://www.gwern.net/GPT-3#prompts-as-programming
- https://learnprompting.org/
- beginner
  - openAI prompt tutorial https://beta.openai.com/docs/quickstart/add-some-examples
  - google LAMDA intro https://aitestkitchen.withgoogle.com/how-lamda-works
  - humanloop prolpt engineering 101 https://website-olo3k29b2-humanloopml.vercel.app/blog/prompt-engineering-101
  - DALLE2 prompt writing book http://dallery.gallery/wp-content/uploads/2022/07/The-DALL%C2%B7E-2-prompt-book-v1.02.pdf
  - https://medium.com/nerd-for-tech/prompt-engineering-the-career-of-future-2fb93f90f117
  - https://wiki.installgentoo.com/wiki/Stable_Diffusion overview
  - https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
  - https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts/
  - https://andymatuschak.org/prompts/
  - for nontechnical
    - https://www.jonstokes.com/p/ai-content-generation-part-1-machine
    - https://www.protocol.com/generative-ai-startup-landscape-map
- Intermediate
  - DALLE2 asset generation + inpainting https://twitter.com/aifunhouse/status/1576202480936886273?s=20&t=5EXa1uYDPVa2SjZM-SxhCQ
  - suhail journey https://twitter.com/Suhail/status/1541276314485018625?s=20&t=X2MVKQKhDR28iz3VZEEO8w
  - composable diffusion - "AND" instead of "and" https://twitter.com/TomLikesRobots/status/1580293860902985728
  - img2img https://andys.page/posts/how-to-draw/
  - quest for photorealism https://www.reddit.com/r/StableDiffusion/comments/x9zmjd/quest_for_ultimate_photorealism_part_2_colors/
    - https://medium.com/merzazine/prompt-design-for-dall-e-photorealism-emulating-reality-6f478df6f186
  - settings tweaking https://www.reddit.com/r/StableDiffusion/comments/x3k79h/the_feeling_of_discovery_sd_is_like_a_great_proc/
    - seed selection https://www.reddit.com/r/StableDiffusion/comments/x8szj9/tutorial_seed_selection_and_the_impact_on_your/
    - minor parameter parameter difference study (steps, clamp_max, ETA, cutn_batches, etc) https://twitter.com/KyrickYoung/status/1500196286930292742
    - Generative AI: Autocomplete for everything https://noahpinion.substack.com/p/generative-ai-autocomplete-for-everything?sd=pf
- Advanced
	- [How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1) 
	- karpathy on transformers
		- https://twitter.com/karpathy/status/1582807367988654081
  - Eugene Yan explanation of the Text to Image stack https://eugeneyan.com/writing/text-to-image/
  - VQGAN/CLIP https://minimaxir.com/2021/08/vqgan-clip/
  - 10 years of Image generation history https://zentralwerkstatt.org/blog/ten-years-of-image-synthesis
  - Vision Transformers (ViT) Explained https://www.pinecone.io/learn/vision-transformers/
  - negative prompting https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/
- https://creator.nightcafe.studio/vqgan-clip-keyword-modifier-comparison VQGAN+CLIP Keyword Modifier Comparison
We compared 126 keyword modifiers with the same prompt and initial image. These are the results.
  - https://creator.nightcafe.studio/collection/8dMYgKm1eVXG7z9pV23W
- Google released PartiPrompts as a benchmark: https://parti.research.google/ "PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as part of this work. P2 can be used to measure model capabilities across various categories and challenge aspects."
- Video tutorials
  - Pixel art https://www.youtube.com/watch?v=UvJkQPtr-8s&feature=youtu.be
- Misc
  - StabilityAI CIO perspective https://danieljeffries.substack.com/p/the-turning-point-for-truly-open?sd=pf
  - https://github.com/awesome-stable-diffusion/awesome-stable-diffusion

## Communities

- StableDiffusion Discord https://discord.com/invite/stablediffusion
- https://reddit.com/r/stableDiffusion
- Akhaliq Discord: https://discord.gg/nYqfg4gnBt
- Deforum Discord https://discord.gg/upmXXsrwZc
- Lexica Discord https://discord.com/invite/bMHBjJ9wRh
- Midjourney's discord
  - how to use midjourney v4 https://twitter.com/fabianstelzer/status/1588856386540417024?s=20&t=PlgLuGAEEds9HwfegVRrpg
- https://stablehorde.net/


## People

This list will be out of date but will get you started. My live list of people to follow is at: https://twitter.com/i/lists/1585430245762441216

- Researchers/Developers
  - https://twitter.com/_jasonwei
  - https://twitter.com/johnowhitaker/status/1565710033463156739
  - https://twitter.com/altryne/status/1564671546341425157
  - https://twitter.com/SchmidhuberAI
  - https://twitter.com/nearcyan
  - https://twitter.com/karinanguyen_
  - https://twitter.com/abhi_venigalla
  - https://twitter.com/advadnoun
  - https://twitter.com/polynoamial
  - https://twitter.com/vovahimself
  - https://twitter.com/sarahookr
  - https://twitter.com/shaneguML
  - https://twitter.com/MaartenSap
  - https://twitter.com/ethanCaballero
  - https://twitter.com/ShayneRedford
  - https://twitter.com/seb_ruder
  - https://twitter.com/rasbt
  - https://twitter.com/wightmanr
  - https://twitter.com/GaryMarcus
  - https://twitter.com/ylecun
  - https://twitter.com/karpathy
  - https://twitter.com/pirroh
  - https://twitter.com/eerac
- News/Aggregators
  - https://twitter.com/ai__pub
  - https://twitter.com/WeirdStableAI
  - https://twitter.com/multimodalart
  - https://twitter.com/LastWeekinAI
  - https://twitter.com/paperswithcode
  - https://twitter.com/DeepLearningAI_
  - https://twitter.com/dl_weekly
  - https://twitter.com/slashML
  - https://twitter.com/_akhaliq
  - https://twitter.com/aaditya_ai
  - https://twitter.com/bentossell
  - https://twitter.com/johnvmcdonnell
- Founders/Builders/VCs
  - https://twitter.com/levelsio
  - https://twitter.com/goodside
  - https://twitter.com/c_valenzuelab
  - https://twitter.com/Raza_Habib496
  - https://twitter.com/sharifshameem/status/1562455690714775552
  - https://twitter.com/genekogan/status/1555184488606564353
  - https://twitter.com/levelsio/status/1566069427501764613?s=20&t=camPsWtMHdSSEHqWd0K7Ig
  - https://twitter.com/amanrsanger
  - https://twitter.com/ctjlewis
  - https://twitter.com/sarahcat21
  - https://twitter.com/jackclarkSF
  - https://twitter.com/alexandr_wang
  - https://twitter.com/rameerez
  - https://twitter.com/scottastevenson
- Stability
  - https://twitter.com/StabilityAI
  - https://twitter.com/StableDiffusion
  - https://twitter.com/hardmaru
  - https://twitter.com/JJitsev
- OpenAI
  - https://twitter.com/sama
  - https://twitter.com/ilyasut
  - https://twitter.com/miramurati
- HuggingFace
  - https://twitter.com/younesbelkada
- Artists
  - https://twitter.com/karenxcheng/status/1564626773001719813
  - https://twitter.com/TomLikesRobots
- Other 
  - Companies
    - https://twitter.com/AnthropicAI
    - https://twitter.com/AssemblyAI
    - https://twitter.com/CohereAI
    - https://twitter.com/MosaicML
    - https://twitter.com/MetaAI
    - https://twitter.com/DeepMind
    - https://twitter.com/HelloPaperspace
- Bots and Apps
  - https://twitter.com/dreamtweetapp
  - https://twitter.com/aiarteveryhour

## Misc

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

## Quotes, Reality & Demotivation

- Narrow, tedium domain usecases https://twitter.com/WillManidis/status/1584900092615528448 and https://twitter.com/WillManidis/status/1584900100480192516
- antihype https://twitter.com/alexandr_wang/status/1573302977418387457
- things stablediffusion struggles with https://opguides.info/posts/aiartpanic/
- New Google
  -  https://twitter.com/alexandr_wang/status/1585022891594510336
-  New Powerpoint
  -  via emad
-  Appending prompts by default in UI
  -  DALLE: https://twitter.com/levelsio/status/1588588688115912705?s=20&t=0ojpGmH9k6MiEDyVG2I6gg

## Infrastructure

- bananadev cold boot problem https://twitter.com/erikdunteman/status/1584992679330426880?s=20&t=eUFvLqU_v10NTu65H8QMbg
- replicate.com
- banana.dev
- huggingface.co
- lambdalabs.com
- astriaAI
- cost of chatgpt - https://twitter.com/tomgoldsteincs/status/1600196981955100694
	- A 3-billion parameter model can generate a token in about 6ms on an A100 GPU
	- a 175b param it should take 350ms secs for an A100 GPU to print out a single word
	- You would need 5 80Gb A100 GPUs just to load the model and text. ChatGPT cranks out about 15-20 words per second. If it uses A100s, that could be done on an 8-GPU server (a likely choice on Azure cloud)
	- On Azure cloud, each A100 card costs about $3 an hour. That's $0.0003 per word generated.
	- The model usually responds to my queries with ~30 words, which adds up to about 1 cent per query.
	- If an average user has made 10 queries per day, I think it’s reasonable to estimate that ChatGPT serves ~10M queries per day.
	- I estimate the cost of running ChatGPT is $100K per day, or $3M per month.

## Legal, Ethics, and Privacy

- NSFW filter https://vickiboykis.com/2022/11/18/some-notes-on-the-stable-diffusion-safety-filter/
- On "AI Art Panic" https://opguides.info/posts/aiartpanic/
- Yannick influencing OPENRAIL-M https://www.youtube.com/watch?v=W5M-dvzpzSQ
- art schools accepting AI art https://twitter.com/DaveRogenmoser/status/1597746558145265664
