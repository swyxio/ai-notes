
## Transcription

- https://github.com/openai/whisper
  - the --initial_prompt CLI arg: For my use, I put a bunch of industry jargon and names that are commonly misspelled in there and that fixes 1/3 to 1/2 of the errors.
  - https://freesubtitles.ai/
  - https://github.com/mayeaux/generate-subtitles
  - [theory](https://twitter.com/ethanCaballero/status/1572692314400628739?s=20&t=j_XtR82eEW6Vp28YvodqJQ): whisper is a way to get more tokens from youtube for gpt4
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

### misc tooling

- https://github.com/words/syllable and ecosystem

### Apps

  - youtube whisper (large-v2 support) https://twitter.com/jeffistyping/status/1600549658949931008

### Translation

- https://github.com/LibreTranslate/LibreTranslate

## Music generation

general consensus is that it's just not very good right now

- disco diffusion?
- https://soundraw.io/ https://news.ycombinator.com/item?id=33727550
