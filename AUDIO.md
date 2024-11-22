
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [Transcription](#transcription)
  - [misc tooling](#misc-tooling)
  - [Apps](#apps)
  - [Translation](#translation)
- [Stem separation](#stem-separation)
- [Music generation](#music-generation)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Transcription (Speech to Text or ASR)

[High level](https://www.reddit.com/r/MachineLearning/comments/14xxg6i/comment/jrsbfps/)

### API

If you simply want to submit your audio files and have an API transcribe them, then Whisper JAX is hands-down the best option for you: [https://huggingface.co/spaces/sanchit-gandhi/whisper-jax](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

The demo is powered by two TPU v4-8's, so it has serious fire-power to transcribe long audio files quickly (1hr of audio in about 30s). It's currently got a limit of 2hr per audio upload, but you could use the Gradio client API to automatically ping this space with all 10k of your 30 mins audio files sequentially, and return the transcriptions: [https://twitter.com/sanchitgandhi99/status/1656665496463495168](https://twitter.com/sanchitgandhi99/status/1656665496463495168)

This way, you get all the benefits of the API, without having to run the model locally yourself! IMO this is the fastest way to set-up your transcription protocol, and also the fastest way to transcribe the audios 😉

https://www.reddit.com/r/MachineLearning/comments/16ftd9v/p_whisper_large_benchmark_137_days_of_audio/ # 137 DAYS of Audio Transcribed in 15 Hours for Just $117 ($0.00059/min)

### Run locally

By locally, we mean running the model yourself (either on your local device, or on a Cloud device). I have experience with a few of these implementations, and here are my thoughts:

1. Original Whisper: [https://github.com/openai/whisper](https://github.com/openai/whisper). Baseline implementation
    
2. Hugging Face Whisper: [https://huggingface.co/openai/whisper-large-v2#long-form-transcription](https://huggingface.co/openai/whisper-large-v2#long-form-transcription). Uses an efficient batching algorithm to give a 7x speed-up on long-form audio samples. By far the easiest way of using Whisper: just `pip install transformers` and run it as per the code sample! No crazy dependencies, easy API, no extra optimisation packages, loads of documentation and love on [GitHub](https://github.com/huggingface/transformers) ❤️. Compatible with fine-tuning if you want this!
    
3. Whisper JAX: [https://github.com/sanchit-gandhi/whisper-jax](https://github.com/sanchit-gandhi/whisper-jax). Builds on the Hugging Face implementation. Written in JAX (instead of PyTorch), where you get a 10x or more speed-up if you run it on TPU v4 hardware (I've gotten up to 15x with large batch sizes for super long audio files). Overall, 70-100x faster than OpenAI if you run it on TPU v4
    
4. Faster Whisper: [https://github.com/guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper). 4x faster than original, also for short form audio samples. But no extra gains for long form on top of this
    
5. Whisper X: [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX). Uses Faster Whisper under-the-hood, so same speed-ups.
    
6. Whisper cpp: [https://github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp). Written in cpp. Super fast to boot up and run. Works on-device (e.g. a laptop or phone) since it's quantised and in cpp. Quoted as transcribing 1hr of audio in approx 8.5 minutes (so about 17x slower than Whisper JAX on TPU v4)

## 2024

- realtime whisper webgpu in browser: https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu
	- june: async version https://huggingface.co/spaces/Xenova/whisper-webgpu

### 2023

- [https://github.com/ochen1/insanely-fast-whisper-cli](https://t.co/sphlCVJ35d)
- [https://github.com/ycyy/faster-whisper-webui](https://t.co/7weHsstQbv)
- [https://github.com/themanyone/whisper_dictation](https://t.co/tyqPlfcADa) 
- [https://github.com/huggingface/distil-whisper](https://t.co/nygkxwiWOt)
- https://pypi.org/project/SpeechRecognition/
- https://github.com/openai/whisper
  - the --initial_prompt CLI arg: For my use, I put a bunch of industry jargon and names that are commonly misspelled in there and that fixes 1/3 to 1/2 of the errors.
  - https://freesubtitles.ai/ (hangs my browser when i try it)
  - https://github.com/mayeaux/generate-subtitles
  - [theory](https://twitter.com/ethanCaballero/status/1572692314400628739?s=20&t=j_XtR82eEW6Vp28YvodqJQ): whisper is a way to get more tokens from youtube for gpt4
  - Real time whisper [https://github.com/shirayu/whispering](https://github.com/shirayu/whispering)
  - whisper running on $300 device https://twitter.com/drjimfan/status/1616471309961269250?s=46&t=4t17Fxog8a65leEnHNZwVw
  - whisper can be hosted on https://deepinfra.com/
  - whisperX with diarization https://twitter.com/maxhbain/status/1619698716914622466 https://github.com/m-bain/whisperX Improved timestamps and speaker identification
	  - model served https://replicate.com/thomasmol/whisper-diarization
	  - https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization
  - real time whisper
	  - https://github.com/davabase/whisper_real_time
	  - https://github.com/openai/whisper/discussions/608
  - whisper as a service self hosting GUI and queueing https://github.com/schibsted/WAAS
  - Live microphone demo (not real time, it still does it in chunks) [https://github.com/mallorbc/whisper_mic](https://github.com/mallorbc/whisper_mic)
  - Whisper webservice ([https://github.com/ahmetoner/whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)) - via this thread
  - Whisper UI https://github.com/hayabhay/whisper-ui
	  - Streamlit UI [https://github.com/hayabhay/whisper-ui](https://github.com/hayabhay/whisper-ui)
	  - Whisper playground [https://github.com/saharmor/whisper-playground](https://github.com/saharmor/whisper-playground)
	  - whisper in the browser https://www.ermine.ai/
  - Transcribe-anything https://github.com/zackees/transcribe-anything automates video fetching and uses whisper to generate .srt, .vtt and .txt files
  - MacWhisper [https://goodsnooze.gumroad.com/l/macwhisper](https://goodsnooze.gumroad.com/l/macwhisper)
  - ios whisper https://whispermemos.com/ 10 free, paid app
  - 🌟Crossplatform desktop Whisper that supports semi-realtime [https://github.com/chidiwilliams/buzz](https://github.com/chidiwilliams/buzz)
  - more whisper tooling https://ramsrigoutham.medium.com/openais-whisper-7-must-know-libraries-and-add-ons-built-on-top-of-it-10825bd08f76
- [https://github.com/dscripka/openWakeWord](https://github.com/dscripka/openWakeWord). The models are readily available in tflite and ONNX formats and are impressively "light" in terms of compute requirements and performance.
- https://github.com/ggerganov/whisper.cpp
  High-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model:
  - Plain C/C++ implementation without dependencies
  - Apple silicon first-class citizen - optimized via Arm Neon and Accelerate framework
  - AVX intrinsics support for x86 architectures
  - Mixed F16 / F32 precision
  - Low memory usage (Flash Attention + Flash Forward)
  - Zero memory allocations at runtime
  - Runs on the CPU
  - C-style API
  - a fork of whisper.cpp that uses DirectCompute to run it on GPUs without Cuda on Windows: https://github.com/Const-me/Whisper
- Whisper.cpp small model is best traadeoff of performance vs accuracy https://blog.lopp.net/open-source-transcription-software-comparisons/
- https://github.com/Vaibhavs10/insanely-fast-whisper Transcribe 150 minutes (2.5 hours) of audio in less than 98 seconds - with OpenAI's Whisper Large v3. 
- Whisper.api - [Open-source, self-hosted speech-to-text with fast transcription](https://github.com/innovatorved/whisper.api)
	- https://news.ycombinator.com/item?id=37226221
-  [https://superwhisper.com](https://superwhisper.com/) is using these whisper.cpp models to provide really good Dictation on macOS.
- Whisper with JAX - 70x faster
	- https://twitter.com/sanchitgandhi99/status/1649046650793648128?s=20
- whisper openai api https://twitter.com/calumbirdo/status/1614826199527690240?s=46&t=-lurfKb2OVOpdzSMz0juIw
- speech separation model https://github.com/openai/whisper/discussions/264#discussioncomment-4706132
	- https://github.com/miguelvalente/whisperer
 - deep speech https://github.com/mozilla/DeepSpeech
	 - out of https://commonvoice.mozilla.org dataset
	 - https://github.com/coqui-ai/TTS fork of deepspeech since 2021
	 - [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech?ref=blog.lopp.net) - an open-source Speech-To-Text engine, using a model trained by machine learning techniques based on Baidu's Deep Speech research paper. It uses Google's TensorFlow to make the implementation easier. Looks like it was actively developed from 2017 to late 2020 but has since been abandoned.
	 - [Flashlight](https://github.com/flashlight/flashlight?ref=blog.lopp.net) is a fast, flexible machine learning library written entirely in C++ from the Facebook AI Research and the creators of Torch, TensorFlow, Eigen and Deep Speech. The project encompasses several apps, including the [Automatic Speech Recognition](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr?ref=blog.lopp.net) app for transcription.
	 - [Speechbrain](https://github.com/speechbrain/speechbrain?ref=blog.lopp.net) is a conversational AI toolkit based on PyTorch. From browsing their documentation it looks like this is more of a programming library designed for building processing pipelines than a standalone transcription tool that you can just feed audio files into. As such, I didn't test it.
- **Deepgram** 80x faster than > Whisper https://news.ycombinator.com/item?id=35367655 - strong endorsement
	- deepgram Nova model https://twitter.com/DeepgramAI/status/1646558003079057409
- Assemblyai conformer https://www.assemblyai.com/blog/conformer-1/
- google has a closed "Universal Speech" model https://sites.research.google/usm/
- whisperspeech - open source TTS 80m model from LAION
	- https://www.youtube.com/watch?v=1OBvf33S77Y

https://news.ycombinator.com/item?id=33663486
-  https://whispermemos.com pressing button on my Lock Screen and getting a perfect transcription in my inbox.
- whisper on AWS - the g4dn machines are the sweet spot of price/performance.
- https://simonsaysai.com to generate subtitles and they had the functionality to input specialized vocabulary,
- https://skyscraper.ai/ using assemblyai
- Read.ai - https://www.read.ai/transcription Provides transcription & diarization and the bot integrates into your calendar. It joins all your meetings for zoom, teams, meet, webex, tracks talk time, gives recommendations, etc.
	- https://huggingface.co/spaces/vumichien/whisper-speaker-diarization This space uses Whisper models from [**OpenAI**](https://github.com/openai/whisper) to recoginze the speech and ECAPA-TDNN model from [**SpeechBrain**](https://github.com/speechbrain/speechbrain) to encode and clasify speakers
	- https://github.com/Majdoddin/nlp pyannote diarization
- https://news.ycombinator.com/item?id=33665692




### Products

- productized whisper https://goodsnooze.gumroad.com/l/macwhisper
	- [whisper turbo](https://whisper-turbo.com) - purely in browser ([tweet context](https://twitter.com/fleetwood___/status/1709364288358662479)), using webgpu
- other speech to text apis
	- rev.com
	- https://text-generator.io/blog/cost-effective-speech-to-text-api
- Podcast summarization
	- feather ai https://twitter.com/joshcadorette/status/1605361535454351362
	- sumly ai https://twitter.com/dvainrub/status/1608175955733798913
- Teleprompter
	- https://github.com/danielgross/teleprompter
		- Everything happens privately on your computer. In order to achieve fast latency locally, we use embeddings or a small fine-tuned model.
		- The data is from Kaggle's quotes database, and the embeddings were computed using SentenceTransformer, which then runs locally on ASR. I also finetuned a small T5 model that sorta works (but goes crazy a lot).
	- https://twitter.com/ggerganov/status/1605322535930941441
- language teacher
	- quazel https://news.ycombinator.com/item?id=32993130
	- https://twitter.com/JavaFXpert/status/1617296705975906306?s=20
- speech to text on the edge https://twitter.com/michaelaubry/status/1635966225628164096?s=20 with arduino nicla voice
- assemblyai conformer-1 https://www.assemblyai.com/blog/conformer-1/
	- https://replit.com/@assemblyai/Speech-To-Text-Example?v=1#main.py

## Text to Speech

https://github.com/Vaibhavs10/open-tts-tracker

- services
	- Play.ht or Podcast.ai - https://arstechnica.com/information-technology/2022/10/fake-joe-rogan-interviews-fake-steve-jobs-in-an-ai-powered-podcast/
		- https://news.ycombinator.com/item?id=35328698#35333601
		- https://news.play.ht/post/introducing-playht-2-0-turbo-the-fastest-generative-ai-text-to-speech-api
	- https://speechify.com/
	- mycroft [https://mycroft.ai/mimic-3/](https://mycroft.ai/mimic-3/)
	- https://blog.elevenlabs.io/enter-the-new-year-with-a-bang/ 
		- https://news.ycombinator.com/item?id=34361651
	- convai -
		- not as flexible, the indian fella at roboflow ai demo wanted to move to elevenlabs
	- murf - a16z ai presentation
	- bigclouds
		- [ https://aws.amazon.com/polly/](https://aws.amazon.com/polly/)
		- [https://cloud.google.com/text-to-speech](https://cloud.google.com/text-to-speech)
		- [https://azure.microsoft.com/en-us/products/cognitive-service...](https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/)
	- Narakeet
		- https://twitter.com/jessicard/status/1642867214943412224
	- https://www.resemble.ai/
	- myshell TTS https://twitter.com/svpino/status/1671488252568834048
- OSS
	- pyttsx3  [https://pyttsx3.readthedocs.io/en/latest/engine.html](https://pyttsx3.readthedocs.io/en/latest/engine.html)
	- https://github.com/lucidrains/audiolm-pytorch Implementation of [AudioLM](https://google-research.github.io/seanet/audiolm/examples/), a Language Modeling Approach to Audio Generation out of Google Research, in Pytorch It also extends the work for conditioning with classifier free guidance with T5. This allows for one to do text-to-audio or TTS, not offered in the paper.
	- tortoise  [https://github.com/neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts)
		- [https://nonint.com/static/tortoise_v2_examples.html](https://nonint.com/static/tortoise_v2_examples.html)
		- used in scribepod https://twitter.com/yacinemtb/status/1608993955835957248?s=46&t=ikA-et-is_MNr-8HTO8e1A
			- https://scribepod.substack.com/p/scribepod-1#details
			- https://github.com/yacineMTB/scribepod/blob/master/lib/processWebpage.ts#L27
	- https://github.com/coqui-ai/TTS
		- previously mozilla TTS
	- [Metavoice TTS - 1b v0.1](https://twitter.com/reach_vb/status/1754984949654904988)
		- includes voice cloning
	- https://github.com/suno-ai/bark
		- tried Bark... at least on CPU-only it's very very slow
		- like 20 seconds to generate a few sentences
	- [dimfeld](https://discord.com/channels/822583790773862470/1154150004437561405/1154169073509351606) likes Mycroft Mimic 3 for locally run, chat assistant usecases that require realtime
	- https://huggingface.co/facebook/mms-tts
	- custom voices
		- https://github.com/neonbjb/tortoise-tts#voice-customization-guide
		- microsoft and google cloud have apis
		- twilio maybe
		- VallE when it comes out
			- https://github.com/Plachtaa/VALL-E-X
	- research papers
		- https://speechresearch.github.io/naturalspeech/
		- research paper from very short voice sample https://valle-demo.github.io/
	- [https://github.com/rhasspy/larynx](https://github.com/rhasspy/larynx)
	- pico2wave with the -l=en-GB flag to get the British lady voice is not too bad for offline free TTS. You can hear it in this video: [https://www.youtube.com/watch?v=tfcme7maygw&t=45s](https://www.youtube.com/watch?v=tfcme7maygw&t=45s)
	- [https://github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) (for very specific non-english purposes, and I was willing to wrangle IPA)
	- Vall-E to synthesize https://twitter.com/DrJimFan/status/1622637578112606208?s=20
		- microsoft?
		- https://github.com/Plachtaa/VALL-E-X
- research unreleased
	- google had something with morgan freeman voice
	- meta voicebox https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/

### misc tooling

- https://github.com/words/syllable and ecosystem
- speaker diarization
	- https://news.ycombinator.com/item?id=33892105
	- [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
	- [https://arxiv.org/abs/2012.00931](https://arxiv.org/abs/2012.00931)
	- example diarization impl https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing
		- from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
	- https://lablab.ai/t/whisper-transcription-and-speaker-identification
- noise cleaning
	- adobe enhance speech for cleaning up spoken audio https://news.ycombinator.com/item?id=34047976 https://podcast.adobe.com/enhance
- https://github.com/elanmart/cbp-translate
	-   Process short video clips (e.g. a single scene)
	-   Work with multiple characters / speakers
	-   Detect and transcribe speech in both English and Polish
	-   Translate the speech to any language
	-   Assign each phrase to a speaker
	-   Show the speaker on the screen
	-   Add subtitles to the original video in a way mimicking the Cyberpunk example
	-   Have a nice frontend
	-   Run remotely in the cloud
- https://essentia.upf.edu/
	- Extensive collection of reusable algorithms
	- Cross-platform
	- Fast prototyping
	- Industrial applications
	- Similarity
	- Classification
	- Deep learning inference
	- Mood detection
	- Key detection
	- Onset detection
	- Segmentation
	- Beat tracking
	- Melody extraction
	- Audio fingerprinting
	- Cover song detection
	- Spectral analysis
	- Loudness metering
	- Audio problems detection
	- Voice analysis
	- Synthesis
- https://github.com/regorxxx/Music-Graph An open source graph representation of most genres and styles found on popular, classical and folk music. Usually used to compute similarity (by distance) between 2 sets of genres/styles.
- https://github.com/regorxxx/Camelot-Wheel-Notation Javascript implementation of the Camelot Wheel, ready to use "harmonic mixing" rules and translations for standard key notations.

### Apps

  - youtube whisper (large-v2 support) https://twitter.com/jeffistyping/status/1600549658949931008
  - list of audio editing ai apps https://twitter.com/ramsri_goutham/status/1592754049719603202?s=20&t=49HqYD7DyViRl_T5foZAxA
  - https://beta.elevenlabs.io/ techmeme ridehome - voice generation in your own voice from existing samples (not reading script)


### Translation

- https://github.com/LibreTranslate/LibreTranslate

## Stem separation

- https://github.com/deezer/spleeter (and bpm detection)
- https://github.com/facebookresearch/demucs demux model - used at outside lands llm ahackathon can strip vocals from a sound https://sonauto.app/
	- used in lalal.ai as well

## Music generation

general consensus is that it's just not very good right now

- Meta https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/
	- AudioCraft consists of three models: [MusicGen](https://huggingface.co/spaces/facebook/MusicGen), [AudioGen](https://felixkreuk.github.io/audiogen/), and [EnCodec](https://ai.meta.com/blog/ai-powered-audio-compression-technique/). 
	- MusicGen, which was trained with Meta-owned and specifically licensed music, generates music from text-based user inputs, 
	- while AudioGen, which was trained on public sound effects, generates audio from text-based user inputs. 
	- Today, we’re excited to release an improved version of 
		- our EnCodec decoder, which allows for higher quality music generation with fewer artifacts; 
		- our pre-trained AudioGen model, which lets you generate environmental sounds and sound effects like a dog barking, cars honking, or footsteps on a wooden floor; and 
		- all of the AudioCraft model weights and code. 
- disco diffusion?
- img-to-music via CLIP interrogator => Mubert ([HF space](https://huggingface.co/spaces/fffiloni/img-to-music), [tweet](https://twitter.com/fffiloni/status/1585698118137483276))
- https://soundraw.io/ https://news.ycombinator.com/item?id=33727550
- Riffusion https://news.ycombinator.com/item?id=33999162
- Bark - text to audio https://github.com/suno-ai/bark
	- https://www.kdnuggets.com/2023/05/bark-ultimate-audio-generation-model.html
- Google AudioLM https://www.technologyreview.com/2022/10/07/1060897/ai-audio-generation/  Google’s new AI can hear a snippet of song—and then keep on playing
	- how it works https://www.shaped.ai/blog/sounding-the-secrets-of-audiolm
- AudioLDM https://github.com/haoheliu/AudioLDM speech, soud effects, music
	- https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation
- MusicLM https://google-research.github.io/seanet/musiclm/examples/
	- reactions https://twitter.com/JacquesThibs/status/1618839343661203456
	- implementation https://github.com/lucidrains/musiclm-pytorch
- https://arxiv.org/abs/2301.12662 singsong voice generation
- small demo apps
	- beatbot.fm https://news.ycombinator.com/item?id=34994444
- sovitz svc - taylor swift etc voice synth
	- https://www.vulture.com/article/ai-singers-drake-the-weeknd-voice-clones.html


## misc

- vocode - ycw23 - 
	- an open source library for building LLM applications you can talk to. Vocode makes it easy to take any text-based LLM and make it voice-based. Our repo is at [https://github.com/vocodedev/vocode-python](https://github.com/vocodedev/vocode-python) and our docs are at [https://docs.vocode.dev](https://docs.vocode.dev/).
	- Building realtime voice apps with LLMs is powerful but hard. You have to orchestrate the speech recognition, LLM, and speech synthesis in real-time (all async)–while handling the complexity of conversation (like understanding when someone is finished speaking or handling interruptions).
	- https://news.ycombinator.com/item?id=35358873
- audio datasets
	- https://github.com/LAION-AI/audio-dataset/blob/main/data_collection/README.md
	- https://www.audiocontentanalysis.org/datasets
	- https://huggingface.co/datasets/Hyeon2/riffusion-musiccaps-dataset/viewer/Hyeon2--riffusion-musiccaps-dataset/train
- audio formats
	- https://github.com/search?q=repo%3Asupercollider%2Fsupercollider++language%3ASuperCollider&type=code
	- https://github.com/search?q=repo%3Agrame-cncm%2Ffaust++language%3AFaust&type=code
	- https://github.com/search?q=repo%3Acsound%2Fcsound++language%3A%22Csound+Document%22&type=code
