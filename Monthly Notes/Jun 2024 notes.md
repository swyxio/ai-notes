
## openai

- nontechnical
	- apple doesnt pay oepnai for chatgpt deal
	- revneue doubels to 3.4b in the last 6 months

## notable reads

- [A Picture is Worth 170 Tokens: How Does GPT-4o Encode Images?](https://www.oranlooney.com/post/gpt-cnn/)
- sigma-gpt https://news.ycombinator.com/item?id=40608413
	- The authors randomly permute (i.e., shuffle) input tokens in training and add two positional encodings to each token: one with the token's position and another with the position of the token to be predicted. Otherwise, the model is a standard autoregressive GPT. The consequences of this seemingly "simple" modification are significant:
	- The authors can prompt the trained model with part of a sequence and then decode the missing tokens, all at once, in parallel, regardless of order -- i.e., the model can in-fill in parallel.
	- The authors can compute conditional probability densities for every missing token in a sequence, again in parallel, i.e., densities for all missing tokens at once.
	- The authors propose a rejection-sampling method for generating in-fill tokens, again in parallel. Their method seems to work well in practice.

## launches

- luma ai dream machine https://news.ycombinator.com/item?id=40670096

## discussions


- forcing AI on to us
	- msft recall default https://news.ycombinator.com/item?id=40610435
- apple intelligence
	- [ipad calculator is nutes](https://x.com/levie/status/1800224021193396594)
	- [talaria tool](https://buttondown.email/ainews/archive/ainews-talaria-apples-new-mlops-superweapon-4066/)
- perplexity - forbes attribution issue
- 