# prompt-eng
notes for prompt engineering


## Stable Diffusion

stable diffusion specific notes

### Major SD forks

Main: https://github.com/CompVis/stable-diffusion

- https://github.com/basujindal/stable-diffusion 
  This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed.
- https://github.com/hlky/stable-diffusion
  - adds a bunch of features - GUI/webui, [textual inversion](https://textual-inversion.github.io/), [upscalers](https://github.com/hlky/stable-diffusion-webui/wiki/Upscalers), mask and crop, img2img editor, word seeds, prompt weighting
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
