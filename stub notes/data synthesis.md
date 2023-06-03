## instruct pix2pix

What's new: Tim Brooks and colleagues at UC Berkeley built InstructPix2Pix, a method that fine-tunes a pretrained text-to-image model to revise images via simple instructions like “swap oranges with bananas” without selecting the area that contained oranges. InstructPix2Pix works with traditional artwork (for which there is no initial prompt) as well as generated images.
Key insight: If you feed an image plus an edit instruction into a typical pretrained image generator, the output may contain the elements you desire but it’s likely to look very different. However, you can fine-tune a pretrained image generator to respond coherently to instructions using a dataset that includes a prompt, an image generated from that prompt, a revised version of the prompt, a corresponding revised version of the image, and an instruction that describes the revision. Annotating hundreds of thousands of images in this way could be expensive, but it’s possible to synthesize such a dataset: (i) Start with a corpus of images and captions, which stand in for prompts. (ii) Use a pretrained large language model to generate revised prompts and instructions. (iii) Then use a pretrained image generator to produce revised images from the revised prompts.


## using RL for data synthesis

https://twitter.com/mathemagic1an/status/1662896309588881408?s=46&t=90xQ8sGy63D2OtiaoGJuww