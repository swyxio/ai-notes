

C4 dataset
- The Multimodal-C4 dataset is an expansion of the text-only [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4), which was used to train [T5 models](https://arxiv.org/abs/1910.10683). For each document in the [C4 en.clean](https://www.tensorflow.org/datasets/catalog/c4#c4en_default_config) dataset, we retrieve the original webpage from [Common Crawl](https://commoncrawl.org/), then collect the downloadable images. Data cleaning is carried out through deduplication and content filtering, which aims to eliminate non-safe for work (NSFW) and unrelated images, such as advertisements. Additionally, we run face detection and discard images with positive identifications. Finally, images and sentences are interleaved using bipartite matching within a document: CLIP ViT/L-14 image-text similarities serve as edge weights. Multimodal-C4 consists of approximately 75 million documents, encompassing around 400M images and 38B tokens. 

contamination
- John graaham cumming and paulg discussion https://twitter.com/jgrahamc/status/1635688698036596763?s=20