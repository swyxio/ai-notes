
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

## Transcription (Speech to Text)

- https://github.com/openai/whisper
  - the --initial_prompt CLI arg: For my use, I put a bunch of industry jargon and names that are commonly misspelled in there and that fixes 1/3 to 1/2 of the errors.
  - https://freesubtitles.ai/
  - https://github.com/mayeaux/generate-subtitles
  - [theory](https://twitter.com/ethanCaballero/status/1572692314400628739?s=20&t=j_XtR82eEW6Vp28YvodqJQ): whisper is a way to get more tokens from youtube for gpt4
  - Buzz transcribes and translates audio offline on your personal computer. Powered by OpenAI's Whisper. 
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


https://news.ycombinator.com/item?id=33663486
-  https://whispermemos.com pressing button on my Lock Screen and getting a perfect transcription in my inbox.
- whisper on AWS - the g4dn machines are the sweet spot of price/performance.
- simonsaysai.com to generate subtitles and they had the functionality to input specialized vocabulary,
- https://skyscraper.ai/ using assemblyai
- https://freesubtitles.ai/
- Read.ai - https://www.read.ai/transcription Provides transcription & diarization and the bot integrates into your calendar. It joins all your meetings for zoom, teams, meet, webex, tracks talk time, gives recommendations, etc.
- https://news.ycombinator.com/item?id=33665692

## Text to Speech

- services
	- Play.ht or Podcast.ai - https://arstechnica.com/information-technology/2022/10/fake-joe-rogan-interviews-fake-steve-jobs-in-an-ai-powered-podcast/
	- mycroft [https://mycroft.ai/mimic-3/](https://mycroft.ai/mimic-3/)
	- 
- OSS
	- pyttsx3  [https://pyttsx3.readthedocs.io/en/latest/engine.html](https://pyttsx3.readthedocs.io/en/latest/engine.html)
	- https://github.com/lucidrains/audiolm-pytorch Implementation of [AudioLM](https://google-research.github.io/seanet/audiolm/examples/), a Language Modeling Approach to Audio Generation out of Google Research, in Pytorch It also extends the work for conditioning with classifier free guidance with T5. This allows for one to do text-to-audio or TTS, not offered in the paper.
	- tortoise  [https://github.com/neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts)
		-  [https://nonint.com/static/tortoise_v2_examples.html](https://nonint.com/static/tortoise_v2_examples.html)
		- used in scribepod https://twitter.com/yacinemtb/status/1608993955835957248?s=46&t=ikA-et-is_MNr-8HTO8e1A
	- [https://github.com/rhasspy/larynx](https://github.com/rhasspy/larynx)
	- pico2wave with the -l=en-GB flag to get the British lady voice is not too bad for offline free TTS. You can hear it in this video: [https://www.youtube.com/watch?v=tfcme7maygw&t=45s](https://www.youtube.com/watch?v=tfcme7maygw&t=45s)
	- [https://github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) (for very specific non-english purposes, and I was willing to wrangle IPA)
	- research paper from very short voice sample https://valle-demo.github.io/

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
	- adobe filter https://news.ycombinator.com/item?id=34047976
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

### Apps

  - youtube whisper (large-v2 support) https://twitter.com/jeffistyping/status/1600549658949931008

### Translation

- https://github.com/LibreTranslate/LibreTranslate

## Stem separation

- https://github.com/deezer/spleeter (and bpm detection)

## Music generation

general consensus is that it's just not very good right now

- disco diffusion?
- img-to-music via CLIP interrogator => Mubert ([HF space](https://huggingface.co/spaces/fffiloni/img-to-music), [tweet](https://twitter.com/fffiloni/status/1585698118137483276))
- https://soundraw.io/ https://news.ycombinator.com/item?id=33727550
- Riffusion https://news.ycombinator.com/item?id=33999162
