
https://forum.effectivealtruism.org/posts/JbScJgCDedXaBgyKC/what-if-we-don-t-need-a-hard-left-turn-to-reach-agi
 [Minerva](https://twitter.com/matthewjbar/status/1542594514589581313?s=21&t=1t02xn9xyriXOte_d3RYyQ) will be considered the most important advance of the year. Minerva is a LLM trained to take math word problems and generate a step-by-step solution. In a particularly impressive result, Minerva got ~50% accuracy on the MATH dataset, much better than the previous 10% state-of-the-art result. This may not sound impressive until you realize how tough the MATH dataset is. This isn't a dataset of 'plug in the numbers' word problems from an introductory algebra class. This dataset is taken from high-school math competitions, designed to challenge the best mathematics students. Estimates of human performance are spotty, but the original MATH dataset paper gave the MATH dataset to a smart Computer Science undergrad student and got a performance around ~40%. Personally, I have a degree in statistics and I'd probably struggle to beat Minerva at these problems. It's conservative to say that Minerva is around 90th percentile of human performance in solving math problems.

https://www.nature.com/articles/d41586-023-00641-w
- To make Minerva, for instance, researchers started with Google’s Pathways Language Model (PaLM), which has 540 billion parameters and was pre-trained on a data set of 780 billion tokens[3](https://www.nature.com/articles/d41586-023-00641-w#ref-CR3). A token can be a word, digit or some unit of information; in PaLM’s case, the tokens were gleaned from English and multilingual web documents, books and code. Minerva was the result of fine-tuning PaLM on tens of billions of tokens from scientific papers and mathematics-related web pages.
- Minerva can answer prompts such as: what is the largest multiple of 30 that is less than 520? The LLM appears to be thinking through the steps, and yet all it is doing is turning the questions into a sequence of tokens, generating a statistically plausible next token, appending it to the original sequence, generating another token, and so on: a process called inference.
- Google researchers fine-tuned three sizes of Minerva using underlying pre-trained PaLM models of 8 billion, 62 billion and 540 billion parameters. Minerva’s performance improved with scale. On the overall MATH data set, the smallest model had 25% accuracy, the medium-sized model reached 43% and the largest breached the 50% mark (see ‘Is a larger chatbot better at mathematics?’).
- The biggest model also used the least amount of fine-tuning data — it was fine-tuned on only 26 billion tokens, whereas the smallest model looked at 164 billion tokens. But the biggest model took a month to fine-tune, on specialized hardware that had eight times as much computing capacity as used for the smallest model, which was fine-tuned for only two weeks. Ideally, the biggest model would have been fine-tuned on more tokens


https://twitter.com/WenhuChen/status/1660832837715611648?s=20
math benchmarks


openai MATH model https://openai.com/research/improving-mathematical-reasoning-with-process-supervision?utm_source=bensbites&utm_medium=newsletter&utm_campaign=openai-s-roadmap

[Llemma: An Open Language Model for Mathematics](https://arxiv.org/abs/2310.10631) ([arxiv.org](https://news.ycombinator.com/from?site=arxiv.org))

anton coq proof model https://github.com/atroyn/math-llm https://twitter.com/atroyn/status/1669111844680900609?s=46&t=90xQ8sGy63D2OtiaoGJuww

## Meta Nougat

- Nougat: an open-source OCR model that accurately scans books with heavy math/scientific notations. It's ages ahead of other open OCR options. Meta is doing extraordinary open-source AI, sometimes without as much fanfare as Llama. https://twitter.com/DrJimFan/status/1702322181928259586
	- use Mathpix instead https://twitter.com/erhartford/status/1713854710375944200