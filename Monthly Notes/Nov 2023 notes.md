## OpenAI
- [Ilya Sutskever on No Priors pod](https://www.youtube.com/watch?v=Ft0gTO2K85A)
- dev day
	- vision
		- https://twitter.com/hellokillian/status/1723106008061587651

### models

- [Yarn-Mistral-7b-128k](https://x.com/mattshumer_/status/1720115354884514042?s=20): 
	- 4x more context than GPT-4. Open-source is the new long-context king! This thing can easily fit entire books in a prompt.
	- [tweet](https://x.com/theemozilla/status/1720107186850877662?s=20)
- Amazon Mistral - 32k
	- https://twitter.com/MatthewBerman/status/1719758392510824505
- Elon X.ai Grok
	- grok vs openchat - https://twitter.com/alignment_lab/status/1721308271946965452


### Fundraising

- Mistral @ 2bn https://archive.ph/hkWD3
- Factory.ai $5m seed https://www.factory.ai/blog https://x.com/matanSF/status/1720106297096593672?s=20
- [Modal labs raised $16m series A with Redpoint](https://twitter.com/modal_labs/status/1711748224610943163?s=12&t=90xQ8sGy63D2OtiaoGJuww)


## other launches

- Elon Musk's X.ai Grok model was announced but not widely released [HN](https://news.ycombinator.com/item?id=38148396)
	- "A unique and fundamental advantage of Grok is that it has real-time knowledge of the world via the X platform."
	- The engine powering Grok is Grok-1, our frontier LLM, which we developed over the last four months. Grok-1 has gone through many iterations over this span of time. [Model Card](https://x.ai/model-card/)
	- [https://x.ai/](https://x.ai/) - Link to waitlist: [https://grok.x.ai/](https://grok.x.ai/) 
	- [UI demo](https://twitter.com/TobyPhln/status/1721053802235621734)
- GitHub Copilot GA
	- https://twitter.com/HamelHusain/status/1723047256180355386
	- workspace https://twitter.com/ashtom/status/1722631796482085227
	- chat https://x.com/ashtom/status/1722330209867993471?s=20
	- mitchell: https://x.com/mitchellh/status/1722346134130348504?s=20
- consistency models
	- krea https://news.ycombinator.com/item?id=38223822

## misc and other discussions

- MFU calculation
	- [Stas Bekman](https://twitter.com/StasBekman/status/1721207940168987113) - This study from 2020 publishes the actual achievable TFLOPS in half-precision for the high-end gpus. e.g., 88% for V100, 93% for A100. Showing that A100@BF16's peak performance is 290 and not 312 TFLOPS. So that means that when we calculate MFU (Model FLOPS Utilization) our reference point shouldn't be the advertised theoretical TFLOPS, but rather the adjusted achievable TFLOPS.
- other good reads
	- [Don't Build AI Products the Way Everyone Else Is Doing It](https://www.builder.io/blog/build-ai) ([builder.io](https://news.ycombinator.com/from?site=builder.io))[103 comments](https://news.ycombinator.com/item?id=38221552)|
