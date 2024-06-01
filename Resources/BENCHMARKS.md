
Benchmarks exist between the Data and Models, and are the least obvious/glamorous but most influential to language model development. They're also an area of intense specialization that the amateurs are only just discovering, and we want to shine a spotlight on them. When you see Chinchilla paper or GPT4 paper, they publish their [MMLU and BIG-bench scores](https://towardsdatascience.com/a-new-ai-trend-chinchilla-70b-greatly-outperforms-gpt-3-175b-and-gopher-280b-408b9b4510) and it bears understanding.

easiest way i know to run the benchmarks yourself is https://github.com/EleutherAI/lm-evaluation-harness
- which was forked from the MMLU test https://huggingface.co/blog/evaluating-mmlu-leaderboard and is also related to the stanford HELM impl
- and drives the Open LLM Leaderboard https://github.com/huggingface/blog/blob/main/open-llm-leaderboard-mmlu.md
openai evals is promising but doesnt have most of them implemented yet


- 1985 [Wordnet](https://en.wikipedia.org/wiki/WordNet)
	- WordNet was first created in 1985, in English only, in the [Cognitive Science](https://en.wikipedia.org/wiki/Cognitive_Science "Cognitive Science") Laboratory of [Princeton University](https://en.wikipedia.org/wiki/Princeton_University "Princeton University") under the direction of [psychology](https://en.wikipedia.org/wiki/Psychology "Psychology") [professor](https://en.wikipedia.org/wiki/Professor "Professor") [George Armitage Miller](https://en.wikipedia.org/wiki/George_Armitage_Miller "George Armitage Miller"). It was later directed by [Christiane Fellbaum](https://en.wikipedia.org/wiki/Christiane_Fellbaum "Christiane Fellbaum").
		- He authored the paper, "[The Magical Number Seven, Plus or Minus Two](https://en.wikipedia.org/wiki/The_Magical_Number_Seven,_Plus_or_Minus_Two "The Magical Number Seven, Plus or Minus Two")," in which he observed that many different experimental findings considered together reveal the presence of an average limit of seven for human [short-term memory](https://en.wikipedia.org/wiki/Short-term_memory "Short-term memory") capacity.
	- The database contains 155,327 words organized in 175,979 [synsets](https://en.wikipedia.org/wiki/Synsets "Synsets") for a total of 207,016 word-sense pairs; in [compressed](https://en.wikipedia.org/wiki/Data_compression "Data compression") form, it is about 12 [megabytes](https://en.wikipedia.org/wiki/Megabyte "Megabyte") in size. It includes the lexical categories [nouns](https://en.wikipedia.org/wiki/Noun "Noun"), [verbs](https://en.wikipedia.org/wiki/Verb "Verb"), [adjectives](https://en.wikipedia.org/wiki/Adjective "Adjective") and [adverbs](https://en.wikipedia.org/wiki/Adverb "Adverb") but ignores [prepositions](https://en.wikipedia.org/wiki/Preposition "Preposition"), [determiners](https://en.wikipedia.org/wiki/Determiner_(linguistics) "Determiner (linguistics)") and other function words. Words from the same lexical category that are roughly synonymous are grouped into [synsets](https://en.wikipedia.org/wiki/Synsets "Synsets"), which include simplex words as well as [collocations](https://en.wikipedia.org/wiki/Collocation "Collocation") like "eat out" and "car pool." The different senses of a [polysemous](https://en.wikipedia.org/wiki/Polysemous "Polysemous") word form are assigned to different synsets. A synset's meaning is further clarified with a short defining _gloss_ and one or more usage examples. An example adjective synset is:
	- ![https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Hamburger_WordNet.png/220px-Hamburger_WordNet.png](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Hamburger_WordNet.png/220px-Hamburger_WordNet.png)
	- All synsets are connected by means of semantic relations. These relations, which are not all shared by all lexical categories, include:
		-   [Nouns](https://en.wikipedia.org/wiki/Noun "Noun")
		    -   _[hypernyms](https://en.wikipedia.org/wiki/Hypernym "Hypernym")_: _Y_ is a hypernym of _X_ if every _X_ is a (kind of) _Y_ (_canine_ is a hypernym of _[dog](https://en.wikipedia.org/wiki/Dog "Dog")_)
		    -   _[hyponyms](https://en.wikipedia.org/wiki/Hyponym "Hyponym")_: _Y_ is a hyponym of _X_ if every _Y_ is a (kind of) _X_ (_dog_ is a hyponym of _canine_)
		    -   _coordinate terms_: _Y_ is a coordinate term of _X_ if _X_ and _Y_ share a hypernym (_wolf_ is a coordinate term of _dog_, and _dog_ is a coordinate term of _wolf_)
		    -   _[meronym](https://en.wikipedia.org/wiki/Meronymy "Meronymy")_: _Y_ is a meronym of _X_ if _Y_ is a part of _X_ (_window_ is a meronym of _building_)
		    -   _[holonym](https://en.wikipedia.org/wiki/Holonymy "Holonymy")_: _Y_ is a holonym of _X_ if _X_ is a part of _Y_ (_building_ is a holonym of _window_)
		-   [Verbs](https://en.wikipedia.org/wiki/Verb "Verb")
		    -   _hypernym_: the verb _Y_ is a hypernym of the verb _X_ if the activity _X_ is a (kind of) _Y_ (_to perceive_ is an hypernym of _to listen_)
		    -   _[troponym](https://en.wikipedia.org/wiki/Troponym "Troponym")_: the verb _Y_ is a troponym of the verb _X_ if the activity _Y_ is doing _X_ in some manner (_to lisp_ is a troponym of _to talk_)
		    -   _[entailment](https://en.wikipedia.org/wiki/Entailment "Entailment")_: the verb _Y_ is entailed by _X_ if by doing _X_ you must be doing _Y_ (_to sleep_ is entailed by _to snore_)
		    -   _coordinate terms_: those verbs sharing a common hypernym (_to lisp_ and _to yell_)
	- There are now WordNets in more than 200 languages.[[4]](https://en.wikipedia.org/wiki/WordNet#cite_note-4)
	- usecases
		- 2. Semantic Similarity: WordNet is used to compute the semantic similarity between words and concepts. This is useful in many applications such as information retrieval, text summarization, and question answering.
		- Lexical Chains: WordNet is used to build lexical chains, which are sequences of related words that occur in a text. This can be useful for text summarization, information retrieval, and text classification.
		- Sentiment Analysis: WordNet is used to identify the sentiment of a text by analyzing the emotion associated with the words. This can be useful for applications such as opinion mining, product reviews, and social media analysis.
		- Machine Translation: WordNet is used to improve the accuracy of machine translation systems by providing information about the meaning and usage of words in different languages.
- 1989 - [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) https://catalog.ldc.upenn.edu/LDC99T42
	- The corpus contains over 4.5 million words of text, including both written and spoken language, and has been annotated with information about syntax, morphology, and other linguistic features. 
	- The Penn Treebank (PTB) project selected 2,499 stories from a three year Wall Street Journal (WSJ) collection of 98,732 stories for syntactic annotation.
- 1998 [MNIST ](https://en.wikipedia.org/wiki/MNIST_database)
	- The MNIST database contains 60,000 training images and 10,000 testing images
	- In 2004, a best-case error rate of 0.42 percent was achieved on the database by researchers using a new classifier called the LIRA, which is a neural classifier with three neuron layers based on Rosenblatt's perceptron principles
	- In 2011, an error rate of 0.27 percent, improving on the previous best result, was reported by researchers using a similar system of neural networks.[[18]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-18) In 2013, an approach based on regularization of neural networks using DropConnect has been claimed to achieve a 0.21 percent error rate.[[19]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-19) In 2016, the single convolutional neural network best performance was 0.25 percent error rate.[[20]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-:0-20) As of August 2018, the best performance of a single convolutional neural network trained on MNIST training data using no [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation "Data augmentation") is 0.25 percent error rate.
	- 2017 - [ EMNIST: an extension of MNIST to handwritten letters](https://arxiv.org/abs/1702.05373)
	- 2017 - [FashionMNIST](https://en.wikipedia.org/wiki/Fashion_MNIST) The dataset contains 70,000 28x28 [grayscale](https://en.wikipedia.org/wiki/Grayscale "Grayscale") images of fashion products from 10 categories from a dataset of [Zalando](https://en.wikipedia.org/wiki/Zalando "Zalando") article images, with 7,000 images per category.
- 2004/9/11/2015 - [Enron Email Dataset](https://en.wikipedia.org/wiki/Enron_Corpus) https://www.cs.cmu.edu/~./enron/
	- The **Enron Corpus** is a database of over 600,000 [emails](https://en.wikipedia.org/wiki/Email "Email") generated by 158 employees[[1]](https://en.wikipedia.org/wiki/Enron_Corpus#cite_note-1) of the [Enron Corporation](https://en.wikipedia.org/wiki/Enron_Corporation "Enron Corporation") in the years leading up to [the company's collapse](https://en.wikipedia.org/wiki/Enron_scandal "Enron scandal") in December 2001. The corpus was generated from Enron email servers by the [Federal Energy Regulatory Commission](https://en.wikipedia.org/wiki/Federal_Energy_Regulatory_Commission "Federal Energy Regulatory Commission") (FERC) during its subsequent investigation.[[2]](https://en.wikipedia.org/wiki/Enron_Corpus#cite_note-2) A copy of the email database was subsequently purchased for $10,000 by [Andrew McCallum](https://en.wikipedia.org/wiki/Andrew_McCallum "Andrew McCallum"), a computer scientist at the [University of Massachusetts Amherst](https://en.wikipedia.org/wiki/University_of_Massachusetts_Amherst "University of Massachusetts Amherst").[[3]](https://en.wikipedia.org/wiki/Enron_Corpus#cite_note-nyt-3) He released this copy to researchers, providing a trove of data that has been used for studies on [social networking](https://en.wikipedia.org/wiki/Social_networking "Social networking") and [computer-mediated communication](https://en.wikipedia.org/wiki/Computer-mediated_communication "Computer-mediated communication").
	- 1.  Email classification: The Enron email dataset has been used to train models to classify emails into different categories, such as spam, personal, or work-related. For example, researchers have used machine learning techniques to train models that can automatically label emails as "urgent" or "non-urgent" based on their content.
	- Email summarization: The Enron email dataset has also been used to develop models that can summarize the content of long email threads into shorter, more concise summaries. This can be useful for quickly understanding the key points of a conversation or for identifying important information buried within a large corpus of emails.
	- Entity recognition: The Enron email dataset has been used to train models that can recognize named entities (such as people, organizations, and locations) within the text of emails. This can be useful for tasks such as identifying the key players in a corporate scandal or tracking the communication patterns of particular individuals or groups.
	- Language modeling: The Enron email dataset has also been used to train large language models, such as LSTMs and Transformers, that can generate new text that resembles the language and style of the original emails. This can be useful for tasks such as generating realistic-looking emails for testing or training chatbots and conversational agents.
- 2006 - 2012 [ImageNet](https://en.wikipedia.org/wiki/ImageNet)
	- AI researcher [Fei-Fei Li](https://en.wikipedia.org/wiki/Fei-Fei_Li "Fei-Fei Li") began working on the idea for ImageNet in 2006. At a time when most AI research focused on models and algorithms, Li wanted to expand and improve the data available to train AI algorithms.[[11]](https://en.wikipedia.org/wiki/ImageNet#cite_note-WiredQuest-11) In 2007, Li met with Princeton professor [Christiane Fellbaum](https://en.wikipedia.org/wiki/Christiane_Fellbaum "Christiane Fellbaum"), one of the creators of [WordNet](https://en.wikipedia.org/wiki/WordNet "WordNet"), to discuss the project. As a result of this meeting, Li went on to build ImageNet starting from the word database of WordNet and using many of its features. 
	- As an assistant professor at Princeton, Li assembled a team of researchers to work on the ImageNet project. They used [Amazon Mechanical Turk](https://en.wikipedia.org/wiki/Amazon_Mechanical_Turk "Amazon Mechanical Turk") to help with the classification of images
	- On 30 September 2012, a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network "Convolutional neural network") (CNN) called [AlexNet](https://en.wikipedia.org/wiki/AlexNet "AlexNet")[[7]](https://en.wikipedia.org/wiki/ImageNet#cite_note-:0-7) achieved a top-5 error of 15.3% in the ImageNet 2012 Challenge, more than 10.8 percentage points lower than that of the runner up.
	- Around 2011, a good ILSVRC classification top-5 error rate was 25%. In 2012, a deep [convolutional neural net](https://en.wikipedia.org/wiki/Convolutional_neural_network "Convolutional neural network") called [AlexNet](https://en.wikipedia.org/wiki/AlexNet "AlexNet") achieved 16%; in the next couple of years, top-5 error rates fell to a few percent.
	- 2014 - [ Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
	- 
- 2009 - [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) Canadian Institute for Advanced Research
	- The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The images were collected from various sources, including the internet and the researchers' personal collections.
	- 2014 - CIFAR 100 - contains 100 classes instead of 10. It was first introduced in 2014 as a more challenging version of the CIFAR-10 dataset.
- 2016 [LAMBADA (Language Model Benchmark for Data-Driven Reasoning)](https://arxiv.org/abs/1606.06031)
	- We introduce LAMBADA, a dataset to evaluate the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative passages sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole passage, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse. We show that LAMBADA exemplifies a wide range of linguistic phenomena, and that none of several state-of-the-art language models reaches accuracy above 1% on this novel benchmark.
	- - In 2016, the LAMBADA benchmark was introduced, and the state-of-the-art language model achieved an accuracy score of 36.5%. 
	- In 2017, Google's Transformer model improved the score to 45.4%. This model introduced the concept of self-attention, which allowed the model to focus on relevant parts of the input sequence, resulting in better accuracy. 
	- In 2018, OpenAI's GPT-2 model achieved a score of 53.4%, which was a significant improvement over the previous year's score. GPT-2 had a larger number of parameters and was trained on a more extensive dataset, allowing it to generate more accurate predictions. 
	- In 2019, Facebook AI's RoBERTa model achieved a score of 64.4%, which was a substantial improvement over the previous year. RoBERTa used a more extensive training dataset and implemented dynamic masking, which improved the model's ability to predict long-term dependencies in the input sequence. RoBERTa also used a larger batch size, which helped to improve the model's training speed and performance.
	- In 2020, Google's T5 model achieved a score of 71.1%, which was the highest score achieved on the LAMBADA benchmark at that time. T5 used a more extensive training dataset and implemented a multitask learning framework, which allowed the model to learn different natural language processing tasks simultaneously. T5 also used a larger model architecture with over 11 billion parameters, which enabled it to generate more accurate predictions.
	- Overall, the major improvements in Large Language Models' scores on the LAMBADA benchmark over time were due to the use of larger model architectures, more extensive training datasets, and the implementation of advanced techniques such as self-attention, dynamic masking, and multitask learning. These improvements have helped to push the boundaries of natural language processing and enable Large Language Models to generate more accurate predictions in various language-related tasks.
- 2018 GLUE (General Language Understanding Evaluation) Introduced in the paper "[GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/pdf/1804.07461.pdf)" by Wang et al. https://gluebenchmark.com/ GLUE consists of nine different datasets, each with a different task
	- SINGLE-SENTENCE TASKS 
		- **CoLA The Corpus of Linguistic Acceptability (Warstadt et al., 2018)** consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.
			- use Matthews correlation coefficient (Matthews, 1975) as the evaluation metric, which evaluates performance on unbalanced binary classification and ranges from -1 to 1, with 0 being the performance of uninformed guessing.
		- **SST-2 The Stanford Sentiment Treebank (Socher et al., 2013)** consists of sentences from movie reviews and human annotations of their sentiment.
			- The task is to predict the sentiment of a given sentence. We use the two-way (positive/negative) class split, and use only sentence-level labels.
	- SIMILARITY AND PARAPHRASE TASKS
		- **MRPC The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005)** is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
			- Because the classes are imbalanced (68% positive), we follow common practice and report both accuracy and F1 score.
		- **QQP The Quora Question Pairs2** dataset is a collection of question pairs from the community question-answering website Quora.
			- The task is to determine whether a pair of questions are semantically equivalent. As in MRPC, the class distribution in QQP is unbalanced (63% negative), so we report both accuracy and F1 score.
		- **STS-B The Semantic Textual Similarity Benchmark (Cer et al., 2017)** is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.
			- Coarse-Grained Categories: 
			- Lexical Semantics
				- Lexical Entailment, Morphological Negation, Factivity, Symmetry/Collectivity, Redundancy, Named Entities, Quantifiers
			- Predicate-Argument Structure
				- Core Arguments, Prepositional Phrases, Ellipsis/Implicits, Anaphora/Coreference Active/Passive, Nominalization, Genitives/Partitives, Datives, Relative Clauses, Coordination Scope, Intersectivity, Restrictivity
			- Logic
				- Negation, Double Negation, Intervals/Numbers, Conjunction, Disjunction, Conditionals, Universal, Existential, Temporal, Upward Monotone, Downward Monotone, Non-Monotone
			- Knowledge
				- Common Sense, World Knowledge
	- INFERENCE TASKS
		- **MNLI The Multi-Genre Natural Language Inference Corpus** (Williams et al., 2018) is a crowdsourced collection of sentence pairs with **textual entailment** annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports.
		- **QNLI (SQUAD) The Stanford Question Answering Dataset** (Rajpurkar et al. 2016) is a question-answering dataset consisting of **question-paragraph pairs**, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). We convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence
		- **RTE The Recognizing Textual Entailment (RTE)** datasets come from a series of annual textual entailment challenges. Combined from 2006-2009 datasets
		- **WNLI The Winograd Schema Challenge (Levesque et al., 2011)** is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence.
			- For example, consider the sentence: The trophy doesn't fit in the brown suitcase because it's too big. In this sentence, the pronoun "it" could either refer to "trophy" or "brown suitcase".
	- ISSUES with GLUE
		- Some critics have pointed out that the GLUE benchmark may not be comprehensive enough, as it only includes a small number of tasks, and
		-  it may be too focused on **sentence-level tasks** rather than more complex tasks that require understanding of longer passages of text.
	- IMPACT
		- The impact of GLUE was evident almost immediately. In the same year, the BERT model achieved a GLUE score of 82.3, which was a considerable improvement over the previous state-of-the-art of 78.4. 
		- Since then, the GLUE score has increased steadily and new models such as RoBERTa, ALBERT, and XLNet have all achieved scores of 90.7 or higher (Dai et al., 2019; Liu et al., 2019; Yang et al., 2019). 
		- In addition to providing a single measure of performance, GLUE also served as a catalyst for further research in NLP. By providing a standard way of evaluating models, researchers have been able to develop new models and compare them to existing ones. This has enabled researchers to focus on the development of more effective models rather than spending time reinventing existing ones.
- 2019 [SuperGLUE (Super General Language Understanding Evaluation)](https://w4ngatang.github.io/static/papers/superglue.pdf)
	- It consists of eight tasks that are more complex than those in GLUE, such as understanding cause-and-effect relationships and commonsense reasoning.
		- The progress of the last twelve months has eroded headroom on the GLUE benchmark dramatically. In the past year, there has been notable progress across many natural language processing (NLP) tasks, led by methods such as ELMo (Peters et al., 2018), OpenAI GPT (Radford et al., 2018), and BERT (Devlin et al., 2019). The common thread connecting these methods is that they couple self-supervised learning from massive unlabelled text corpora with a recipe for effectively adapting the resulting model to target tasks.
		- While some tasks (Figure 1) and some linguistic phenomena (Figure 2 in Appendix B) measured in GLUE remain difficult, the current state of the art GLUE Score as of early July 2019 (88.4 from Yang et al., 2019) surpasses human performance (87.1 from Nangia and Bowman, 2019) by 1.3 points, and in fact exceeds this human performance estimate on four tasks. Consequently, while there remains substantial scope for improvement towards GLUE’s high-level goals, the original version of the benchmark is no longer a suitable metric for quantifying such progress.
	- However, it improves upon GLUE in several ways: • 
		- More challenging tasks: SuperGLUE retains the two hardest tasks in GLUE. The remaining tasks were identified from those submitted to an open call for task proposals and were selected based on difficulty for current NLP approaches. • 
		- More diverse task formats: The task formats in GLUE are limited to sentence- and sentence-pair classification. We expand the set of task formats in SuperGLUE to include coreference resolution and question answering (QA). • 
		- Comprehensive human baselines: We include human performance estimates for all benchmark tasks, which verify that substantial headroom exists between a strong BERT-based baseline and human performance. • 
		- Improved code support: SuperGLUE is distributed with a new, modular toolkit for work on pretraining, multi-task learning, and transfer learning in NLP, built around standard tools including PyTorch (Paszke et al., 2017) and AllenNLP (Gardner et al., 2017). • 
		- Refined usage rules: The conditions for inclusion on the SuperGLUE leaderboard have been revamped to ensure fair competition, an informative leaderboard, and full credit assignment to data and task creators.
	- tasks
		- BoolQ (Boolean Questions, Clark et al., 2019) is a QA task where each example consists of a short passage and a yes/no question about the passage. The questions are provided anonymously and unsolicited by users of the Google search engine, and afterwards paired with a paragraph from a Wikipedia article containing the answer. Following the original work, we evaluate with accuracy. 
			- Passage: Barq’s – Barq’s is an American soft drink. Its brand of root beer is notable for having caffeine. Barq’s, created by Edward Barq and bottled since the turn of the 20th century, is owned by the Barq family but bottled by the Coca-Cola Company. It was known as Barq’s Famous Olde Tyme Root Beer until 2012. Question: is barq’s root beer a pepsi product Answer: No
		- CB (CommitmentBank, De Marneffe et al., 2019) is a corpus of short texts in which at least one sentence contains an embedded clause. Each of these embedded clauses is annotated with the degree to which it appears the person who wrote the text is committed to the truth of the clause.
			- Text: B: And yet, uh, I we-, I hope to see employer based, you know, helping out. You know, child, uh, care centers at the place of employment and things like that, that will help out. A: Uh-huh. B: What do you think, do you think we are, setting a trend? Hypothesis: they are setting a trend Entailment: Unknown
		- COPA (Choice of Plausible Alternatives, Roemmele et al., 2011) is a causal reasoning task in which a system is given a premise sentence and must determine either the cause or effect of the premise from two possible choices.
				-  Correct Alternative: 1
		- MultiRC (Multi-Sentence Reading Comprehension, Khashabi et al., 2018) is a QA task where each example consists of a context paragraph, a question about that paragraph, and a list of possible answers. The system must predict which answers are true and which are false.
			- Paragraph: Susan wanted to have a birthday party. She called all of her friends. She has five friends. Her mom said that Susan can invite them all to the party. Her first friend could not go to the party because she was sick. Her second friend was going out of town. Her third friend was not so sure if her parents would let her. The fourth friend said maybe. The fifth friend could go to the party for sure. Susan was a little sad. On the day of the party, all five friends showed up. Each friend had a present for Susan. Susan was happy and sent each friend a thank you card the next week 
			- Question: Did Susan’s sick friend recover? Candidate answers: Yes, she recovered (T), No (F), Yes (T), No, she didn’t recover (F), Yes, she was at Susan’s party (T)
		- The set of eight tasks in our benchmark emphasizes diverse task formats and low-data training data tasks, with nearly half the tasks having fewer than 1k examples and all but one of the tasks having fewer than 10k examples. We evaluate BERT-based baselines and find that they still lag behind humans by nearly 20 points. Given the difficulty of SuperGLUE for BERT, we expect that further progress in multi-task, transfer, and unsupervised/self-supervised learning techniques will be necessary to approach human-level performance on the benchmark. Overall, we argue that SuperGLUE offers a rich and challenging testbed for work developing new general-purpose machine learning methods for language understanding.
	- Issues: Some have criticized SuperGLUE for being too difficult, which makes it hard to achieve high scores even for state-of-the-art models. There are also concerns that it may not be representative of real-world language understanding tasks.
	- Impact: The impact of the SuperGLUE benchmarks on LLM development has been significant, as it has pushed researchers to develop more advanced models and techniques to achieve state-of-the-art performance on these tasks. 
		- Specifically, the benchmarks have led to the development of models that incorporate advanced techniques such as pre-training, fine-tuning, and attention mechanisms to improve their performance on natural language understanding tasks.
		- For example, the T5 model, which was introduced in 2019, achieved state-of-the-art performance on several SuperGLUE tasks, including natural language inference, question answering, and reading comprehension. The T5 model uses a pre-training approach that involves training a large transformer model on a diverse set of tasks and then fine-tuning it on specific tasks within the SuperGLUE benchmark. This approach allows the model to learn more about the structure and semantics of natural language and to generalize better to new tasks. 
		- Other models that have achieved significant improvements on the SuperGLUE benchmarks include RoBERTa, which uses a similar pre-training and fine-tuning approach to T5, and BERT-large, which incorporates a larger transformer model and attention mechanisms to improve its performance on natural language understanding tasks.
- 2019 [CoQA Conversational Question Answering Dataset](https://arxiv.org/abs/1808.07042)
	- https://stanfordnlp.github.io/coqa/
	- CoQA is a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation. 
	- CoQA contains 127,000+ questions with answers collected from 8000+ conversations. Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. The unique features of CoQA include 1) the questions are conversational; 2) the answers can be free-form text; 3) each answer also comes with an evidence subsequence highlighted in the passage; and 4) the passages are collected from seven diverse domains. CoQA has a lot of challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning.
	- We evaluate strong dialogue and reading comprehension models on CoQA. The best system obtains an F1 score of 65.4%, which is 23.4 points behind human performance (88.8%), indicating there is ample room for improvement. https://stanfordnlp.github.io/coqa/
	- Example
		- Jessica went to sit in her rocking chair. Today was her birthday and she was turning 80. Her granddaughter Annie was coming over in the afternoon and Jessica was very excited to see her. Her daughter Melanie and Melanie’s husband Josh were coming as well. Jessica had . . . 
		- Q1: Who had a birthday? A1: Jessica R1: Jessica went to sit in her rocking chair. Today was her birthday and she was turning 80. 
		- Q2: How old would she be? A2: 80 R2: she was turning 80 
		- Q3: Did she plan to have any visitors? A3: Yes R3: Her granddaughter Annie was coming over 
		- Q4: How many? A4: Three R4: Her granddaughter Annie was coming over in the afternoon and Jessica was very excited to see her. Her daughter Melanie and Melanie’s husband Josh were coming as well. 
		- Q5: Who? A5: Annie, Melanie and Josh R5: Her granddaughter Annie was coming over in the afternoon and Jessica was very excited to see her. Her daughter Melanie and Melanie’s husband Josh were coming as well.
- 2019-10 AI2 Reasoning Challenge https://leaderboard.allenai.org/arc/submissions/about
	- Think you have solved question answering? Try the AI2 Reasoning Challenge (ARC)! The ARC dataset contains 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. They are drawn from a variety of sources, and sorted into a Challenge Set of 2590 “hard” questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5197 questions, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. This leaderboard is for the CHALLENGE Set.
	- https://arxiv.org/abs/1803.05457
	- The Allen Institute for Artificial Intelligence (AI2) is involved in the development of large language model (LLM) benchmarks because it is a research organization dedicated to advancing artificial intelligence (AI) and natural language processing (NLP) technologies.
		- 1.  Need for challenging benchmarks:
		1.  Focus on real-world applications
		3.  Collaboration with the research community: T
	- A student riding a bicycle observes that it moves faster on a smooth road than on a rough road. This happens because the smooth road has (A) less gravity (B) more gravity (C) less friction [correct] (D) more friction
	- GPT 3.5 - 85.2
	- GPT-4 - 96.3%
- 2019 - WinoGrande: An Adversarial Winograd Schema Challenge at Scale https://winogrande.allenai.org/ Keisuke Sakaguchi, et al Washington https://arxiv.org/abs/1907.10641
	- The Winograd Schema Challenge (WSC) (Levesque, Davis, and Morgenstern 2011), a benchmark for commonsense reasoning, is a set of **273 expert-crafted pronoun resolution** problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations. The benchmark is divided into two main parts: WinoGrande 1.0, which contains questions that require understanding of linguistic context, and WinoGrande 2.0, which contains questions that require understanding of situational context.
	- **WinoGrande**, a large-scale dataset of 44k problems, 
	- The best state-of-the-art methods on WinoGrande achieve 59.4 – 79.1%, which are 15 - 35% (absolute) below human performance of 94.0%, depending on the amount of the training data allowed (2% – 100% respectively).
	- previously: 
		The trophy doesn't fit in the brown suitcase because it's too big. In this sentence, the pronoun "it" could either refer to "trophy" or "brown suitcase".
	- now
		- Sarah was a much better surgeon than Maria so _ always got the easier cases.
		- Jen made charcoal to use as a mask and as toothpaste, but using the _ made her skin very black.	mask	toothpaste
	- Roberta - 0.70s
	- T5 - 0.85s
	- GPT4 - 87.5
- 2018. [Swag: A large-scale adversarial dataset for grounded commonsense inference.](https://arxiv.org/abs/1808.05326)
	- "Situations With Adversarial Generations"
	- Given a partial description like “she opened the hood of the car,” humans can reason about the situation and anticipate what might come next (“then, she examined the engine”). In this paper, we introduce the task of grounded commonsense inference, unifying natural language inference and commonsense reasoning. We present Swag, a new dataset with 113k multiple choice questions about a rich spectrum of grounded situations.
	- When the SWAG dataset was first announced (Zellers et al., 2018), this new task of commonsense natural language inference seemed trivial for humans (88%) and yet challenging for thenstate-of-the-art models (ă60%), including ELMo (Peters et al., 2018). However, BERT (Devlin et al., 2018) soon reached over 86%, almost human-level performance. One news article on this development was headlined “finally, a machine that can finish your sentence.”
	- On stage, a woman takes a seat at the piano. She a) sits on a bench as her sister plays with the doll. b) smiles with someone as the music plays. c) is in the crowd, watching the dancers. **d) nervously sets her fingers on the keys.** 
- 2019 [HellaSwag](https://rowanzellers.com/hellaswag/) - Can a Machine _Really_ Finish Your Sentence? https://arxiv.org/abs/1905.07830 -   "Commonsense reasoning around everyday events"
	- The benchmark is designed to evaluate the ability of natural language processing (NLP) models to handle context-dependent language understanding tasks that require common-sense reasoning and world knowledge.
	- The HellaSwag benchmark consists of a set of multiple-choice questions that are designed to be difficult for NLP models to solve through simple pattern recognition or syntactic parsing. The questions require a deeper understanding of context and require common-sense reasoning and world knowledge to answer correctly.
	- drawing from WikiHow and ActivityNet. 5000 indomain, 5000 zeroshot
	- Random 25
	- GPT 41
	- BERT 47
	- Grover 57-75
	- Roberta 85
	- GPT3.5 85
	- GPT4 95
	- "36% of hellaswag contains errors" https://old.reddit.com/r/LanguageTechnology/comments/zkia03/36_of_hellaswag_benchmark_contains_errors/
		- https://www.surgehq.ai/blog/hellaswag-or-hellabad-36-of-this-popular-llm-benchmark-contains-errors?utm_source=ainews&utm_medium=email&utm_campaign=ainews-132024-rip-coqui
- 2019 DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs https://arxiv.org/pdf/1903.00161.pdf
	- The DROP benchmark consists of a set of questions that require discrete reasoning over paragraphs of text.
	- That year, his Untitled (1981), a painting of a haloed, black-headed man with a bright red skeletal body, depicted amid the artists signature scrawls, was sold by Robert Lehrman for $16.3 million, well above its $12 million high estimate. 
		- How many more dollars was the Untitled (1981) painting sold for than the 12 million dollar estimation? 
		- 4300000
	- Before the UNPROFOR fully deployed, the HV clashed with an armed force of the RSK in the village of Nos Kalik, located in a pink zone near Sibenik, and captured ˇ the village at 4:45 p.m. on 2 March 1992. The JNA formed a battlegroup to counterattack the next day.
		- What date did the JNA form a battlegroup to counterattack after the village of Nos Kalik was captured?
		- 3 March 1992
	- GPT 3.5 - 64%
	- GPT-4 - 81%
- 2020 XTREME -  X-Lingual Transfer Evaluation of Multilingual Encoders
	- https://sites.research.google/xtreme
	- To encourage more research on multilingual transfer learning, we introduce the Cross-lingual TRansfer Evaluation of Multilingual Encoders (XTREME) benchmark. XTREME covers 40 typologically diverse languages spanning 12 language families and includes 9 tasks that require reasoning about different levels of syntax or semantics.
	- The languages in XTREME are selected to maximize language diversity, coverage in existing tasks, and availability of training data. Among these are many under-studied languages, such as the Dravidian languages Tamil (spoken in southern India, Sri Lanka, and Singapore), Telugu and Malayalam (spoken mainly in southern India), and the Niger-Congo languages Swahili and Yoruba, spoken in Africa.
	- The impact of XTREME 
		- can be seen in the rapid progress that has been made in the development of cross-lingual language models since its introduction. Researchers and developers have used the benchmark to evaluate and improve their models, leading to significant advances in areas such as transfer learning, multilingual modeling, and cross-lingual understanding. 
		- Many of the state-of-the-art cross-lingual language models, such as XLM-R and mBERT, have been evaluated on XTREME and have achieved high scores.
		- Moreover, the XTREME benchmark has also inspired the development of new models that specifically address cross-lingual transfer, such as mT5 and XLSR, which aim to improve upon the performance of existing models on the XTREME benchmark. Overall, the impact of the XTREME benchmark has been to drive progress in the development of more accurate, robust, and versatile cross-lingual language models that can better understand and process natural language across multiple languages.
- 2021- [HumanEval - Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
	- We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot.
	- To accurately benchmark our model, we create a dataset of 164 original programming problems with unit tests. These problems assess language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions. We release this data along with an evaluation framework at https://www.github.com/openai/human-eval.
	- repeated sampling - solves 70% with 100 samples
	- code examples
		- def incr_list(l: list):
		- def solution(list): 
		- def encode_cyclic, def decode_cyclic
		- docstring generation
	- Codex - 28.8% of the problems
	- GPT-J solves 11.4%
	- GPT3.5 - 48%
	- GPT4 - 67%
- 2021 - [MMLU - Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
	- https://github.com/hendrycks/test
	- a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average.
		- These include practice questions for tests such as the Graduate Record Examination and the United States Medical Licensing Examination. It also includes questions designed for undergraduate courses and questions designed for readers of Oxford University Press books. Some tasks cover a subject, like psychology, but at a specific level of difficulty, such as “Elementary,” “High School,” “College,” or “Professional.” For example, the “Professional Psychology” task draws on questions from freely available practice questions for the Examination for Professional Practice in Psychology, while the “High School Psychology” task has questions like those from Advanced Placement Psychology examinations.
	- Examples from the Microeconomics task. 
		- One of the reasons that the government discourages and regulates monopolies is that (A) producer surplus is lost and consumer surplus is gained. (B) monopoly prices ensure productive efficiency but cost society allocative efficiency. (C) monopoly firms do not engage in significant research and development. (D) **consumer surplus is lost with higher prices and lower levels of output.**
	- Examples from the Conceptual Physics and College Mathematics STEM tasks
		- When you drop a ball from rest it accelerates downward at 9.8 m/s². If you instead throw it downward assuming no air resistance its acceleration immediately after leaving your hand is **(A) 9.8 m/s²** (B) more than 9.8 m/s² (C) less than 9.8 m/s² (D) Cannot say unless the speed of throw is given.
		- In the complex z-plane, the set of points satisfying the equation z² = |z|² is a (A) pair of points (B) circle (C) half-line **(D) line**
	- professional medicine
		- A 33-year-old man undergoes a radical thyroidectomy for thyroid cancer. During the operation, moderate hemorrhaging requires ligation of several vessels in the left side of the neck. Postoperatively, serum studies show a calcium concentration of 7.5 mg/dL, albumin concentration of 4 g/dL, and parathyroid hormone concentration of 200 pg/mL. Damage to which of the following vessels caused the findings in this patient? (A) Branch of the costocervical trunk (B) Branch of the external carotid artery **(C) Branch of the thyrocervical trunk** (D) Tributary of the internal jugular vein
	- Random Baseline 25
	- GPT2 - 32
	- GPT3 - 43 - 60
	- Gopher 280B - 60
	- Chinchilla 70B - 67.5
	- GPT3.5 - 70
	- GPT4 - 86.4 https://twitter.com/DanHendrycks/status/1635706824308719617
		- Since it gets 86.4% on our MMLU benchmark, that suggests GPT-4.5 should be able to reach expert-level performance.
- 2022 - BIG-bench
	- https://www.deeplearning.ai/the-batch/toward-next-gen-language-models/
	- https://arxiv.org/pdf/2206.04615.pdf
	- https://cs.nyu.edu/~davise/Benchmarks/BIG-bench.html
	- BIG-bench currently consists of 204 tasks, contributed by 442 authors across 132 institutions. Task topics are diverse, drawing problems from linguistics, childhood development, math, common-sense reasoning, biology, physics, social bias, software development, and beyond.
	- The authors selected over 200 [tasks](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/README.md?ref=the-batch-deeplearning-ai) based on 10 [criteria](https://github.com/google/BIG-bench/blob/main/docs/doc.md?ref=the-batch-deeplearning-ai) such as being sensible to humans, not solved by current language models, and “not solvable by memorizing the internet.” Many involve atypical problems such as identifying a single move that will win a game of chess, guessing a movie title from a series of emojis, and playing a role in a mock courtroom trial.
	- For example, answering multiple-choice questions about Hindu mythology, the best model scored around 76 percent, the average human scored roughly 61 percent, and the best human scored 100 percent (random chance was 25 percent). Generally, larger models performed better than smaller ones. For example, BIG-G’s average accuracy on three-shot, multiple-choice tasks was nearly 33 percent with a few million parameters but around 42 percent with over a hundred billion parameters.
	- including
		- chess "In the following chess position, find a checkmate-in-one move."
		- Q: What is the name of the element with an atomic number of 6? A:
		- Identify whether a sentence involves an anachronism.
			-   "During the Allied bombardment of the beaches of Iwo Jima, Ralph spoke loudly into his radio."  
			- "During the Allied bombardment of the beaches of Iwo Jima, Ralph spoke loudly into his iPhone."
		- What movie does this emoji describe? 👧🐟🐠🐡 Finding Nemo
		- What movie does this emoji describe? 🦸🦸‍♂️👦👧👶Incredibles
		- https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/
- 2023 - Stanford HELM
	- https://crfm.stanford.edu/helm/v0.2.0/?group=core_scenarios
	- https://twitter.com/percyliang/status/1610373246477819904
	- evaluates 34 prominent language models in a standardized way on 42 scenarios x 7 metrics.
		- https://crfm.stanford.edu/helm/latest/?groups=1
	- OpenAI 1, anthropic 2, then ... then cohere, bigscience bloom, meta OPT
	- text-davinci-002 does WORSE than text-davinci-003
		- text-davinci-003 improves over text-davinci-002 on 10/16 core scenarios but underperforms significantly on IMDB (82.4% versus 94.6%), causing it to be ranked lower.
		- Digging in a bit deeper, text-davinci-003 outputs an invalid category “Neutral” for some reviews despite being prompted with only “Positive” and “Negative” in the in-context examples. It seems to have a stronger prior that is harder to override.
- 2023 - Multi-PLE benchmark
	- used in codellama paper
	- "we evaluate our models on a more diverse set of programming languages. For that, we use the MultiPL-E benchmark (Cassano et al., 2023). We report results for Python, C++, Java, PHP, TypeScript, C#, and Bash in Table 4.""

### 2023 - GPT-4

https://openai.com/research/gpt-4
- Bar exam
- LSAT
- SAT
- GREs
- AP Bio
- AMC
- Sommelier
- Leetcode

1.  Purpose: The AMC 10 and AMC 12 are both designed to test a student's knowledge of high school mathematics and their problem-solving skills. The AMC 10 is intended for students in grades 9 and 10, while the AMC 12 is intended for students in grades 11 and 12. Both competitions cover a wide range of topics in algebra, geometry, number theory, and combinatorics.
2. 1.  Format: The AMC 10 and AMC 12 are both multiple-choice exams that consist of 25 questions. Students are given 75 minutes to complete the exam, and each correct answer is worth 6 points, each incorrect answer is worth 1.5 points, and unanswered questions receive 0 points.
2.  Scoring: The AMC 10 and AMC 12 are scored on a scale from 0 to 150, with a minimum score of 0 and a maximum score of 150. The scores are based on the number of correct answers, with partial credit given for incorrect answers that show some work or reasoning.
3. AMC 10 - 30/150
4. AMC 12 - 60/150

https://twitter.com/WenhuChen/status/1660832837715611648?s=20
We propose TheoremQA, the first theorem-driven dataset to evaluate LLMs' capabilities to solve theorem-driven questions in Math, Physics, EE&CS and Finance. We carefully curate 800 questions covering 350+ theorems across 30+ subfields. Each question is accompanied with a theorem.

## Selected Concepts


### Textual entailment

**Textual entailment** is a natural language processing task that involves determining whether one piece of text (the "premise") logically entails another piece of text (the "hypothesis"). In other words, given a premise and a hypothesis, the task is to determine whether the premise implies or entails the hypothesis. Here are some examples:

- Example 1: Premise: The cat is sleeping on the couch. Hypothesis: The animal is resting. 
	- Entailment: True 
	- Explanation: The premise implies the hypothesis, because if the cat is sleeping on the couch, it is resting.
- Example 2: Premise: The dog barked loudly all night. Hypothesis: The dog slept soundly all night. 
	- Entailment: False 
	- Explanation: The premise contradicts the hypothesis, because if the dog barked loudly all night, it likely did not sleep soundly.
- Example 3: Premise: The movie received rave reviews from critics. Hypothesis: Many people disliked the movie. 
	- Entailment: False 
	- Explanation: The premise does not imply the hypothesis, because just because critics liked the movie does not mean that many people disliked it.

Textual entailment is a challenging task for natural language processing models because it requires understanding the meaning and context of the text, as well as the relationships between words and phrases.

Causal reasoning benchmark - Tubingen https://twitter.com/amt_shrma/status/1653457982062219266?s=20

### World Knowledge 

In this category we focus on knowledge that can clearly be expressed as facts, as well as broader and less common geographical, legal, political, technical, or cultural knowledge. 
-  “This is the most oniony article I’ve seen on the entire internet” entails “This article reads like satire”. 
- “The reaction was strongly exothermic” entails “The reaction media got very hot”. 
- “There are amazing hikes around Mt. Fuji” entails “There are amazing hikes in Japan” but not “There are amazing hikes in Nepal”. 

### Common Sense 

In this category we focus on knowledge that is more difficult to express as facts and that we expect to be possessed by most people independent of cultural or educational background. This includes a basic understanding of physical and social dynamics as well as lexical meaning (beyond simple lexical entailment or logical relations). 

- “The announcement of Tillerson’s departure sent shock waves across the globe” contradicts “People across the globe were prepared for Tillerson’s departure”.
- “Marc Sims has been seeing his barber once a week, for several years” entails “Marc Sims has been getting his hair cut once a week, for several years”. 
- “Hummingbirds are really attracted to bright orange and red (hence why the feeders are usually these colours)” entails “The feeders are usually coloured so as to attract hummingbirds”.


### Contamination

The problem of benchmark contamination in the context of large language model (LLM) development refers to the issue of models achieving artificially high scores on benchmark datasets by memorizing or overfitting to specific examples in the dataset rather than truly understanding the underlying task. This can lead to inflated claims about a model's performance and may not reflect its actual performance in real-world applications. To address this issue, researchers have proposed new benchmarks, such as the GLUE and SuperGLUE benchmarks, that evaluate models on a wider range of tasks to reduce the risk of overfitting.

- Leetcode - I just tested GPT-4 code output using a Leetcode test. I choose a more challenging example. GPT-4 solved it with-in 5 seconds and placed it in the #1 spot for execution time out 975K entries. https://twitter.com/raymonnnddd/status/1636159008183558144
- Of the easiest problems on Codeforces, it solved 10/10 pre-2021 problems and 0/10 recent problems. https://twitter.com/cHHillee/status/1635790330854526981
	- https://www.aisnakeoil.com/p/gpt-4-and-professional-benchmarks

## Other benchmarking issues

### Bias

1.  Bias: Benchmark datasets may contain biases that are not representative of the real world or that unfairly advantage or disadvantage certain models. For example, a dataset may contain a disproportionate number of examples from a particular demographic group or language, which could lead to biased results.
	1. Imagenet: Specifically, the ImageNet dataset contains many images of people that are labeled based on their occupation, activity, or context (such as "businessman", "cheerleader", or "homeless person"). However, the labels often include racial or ethnic identifiers, such as "Asian businessman" or "African American cheerleader". This can reinforce harmful stereotypes and perpetuate biases against certain racial or ethnic groups.

### Data quality

2.  Data quality: Benchmark datasets may contain errors or inconsistencies that can affect the accuracy of the evaluation. For example, a dataset may contain mislabeled examples or instances where the ground truth is unclear or disputed.
	1. 1988 - The Petals dataset, also known as the Iris dataset, is a classic dataset in machine learning that is often used for classification tasks. It contains measurements of the sepal length, sepal width, petal length, and petal width of three different species of iris flowers: Setosa, Versicolor, and Virginica.
	2. The data quality issue with the Petals dataset is that it contains a number of errors in the measurements of one of the species, the iris Versicolor. Specifically, there are a small number of instances where the petal length is shorter than the petal width, which is biologically impossible. This error is thought to have been introduced during the data collection process.

### Task specificity

3.  Task specificity: Benchmark datasets may be too specific to a particular task or domain, which can make it difficult to evaluate a model's overall performance or generalization ability. For example, a dataset that focuses only on one type of language task, such as sentiment analysis, may not be representative of the full range of language understanding tasks.
	1. SQuAD (Stanford Question Answering Dataset
	2. Specifically, the SQuAD dataset contains a large number of questions that are based on textual evidence within a single paragraph, which is a relatively narrow type of question answering task. While this task is useful for evaluating models that can extract information from a specific text passage, it may not be representative of the full range of question answering tasks that are encountered in real-world applications.
	3. For example, in many real-world question answering tasks, the relevant information may be spread across multiple documents or sources, or the question may require a more complex form of reasoning or inference.

### Reproducibility

4.  Reproducibility: Benchmarking results may be difficult to reproduce or compare across different models or research groups, which can make it challenging to assess progress in the field. For example, different research groups may use different preprocessing methods or evaluation metrics, which can make it difficult to compare results across studies.
	1. Specifically, the MNIST dataset is often preprocessed by centering and scaling the pixel values of the images to have zero mean and unit variance. However, there is no universally agreed-upon way to perform this normalization, and different researchers may use slightly different methods or parameters. This can lead to inconsistencies in the evaluation of machine learning models trained on the dataset, as different models may be evaluated using different preprocessing methods.

### Resource requirements

5.  Resource requirements: Benchmark datasets may require significant computational resources or specialized hardware to evaluate, which can limit the accessibility of the evaluation to certain researchers or organizations.
	1. eg. GPT3/4

### calibrating confidence

6. calibrating confidence
	1. GPT-4 can also be confidently wrong in its predictions, not taking care to double-check work when it’s likely to make a mistake. Interestingly, the base pre-trained model is highly calibrated (its predicted confidence in an answer generally matches the probability of being correct). However, through our current post-training process, the calibration is reduced.





## games as benchmarks

- poker and diplomacy https://overcast.fm/+_C9fBMhI4
- before that it was Go, Starcraft, chess
- important to pre exist before researchers so that have human baseline and not artificial 



## Misc Ungroomed Notes


text
- https://www.deeplearning.ai/the-batch/issue-176/ Answering the call, more than 130 institutions collaborated on [BIG-bench](https://www.deeplearning.ai/the-batch/toward-next-gen-language-models/), which includes tasks like deducing a movie title from emojis, participating in mock trials, and detecting logical fallacies.
- https://cs.nyu.edu/~davise/Benchmarks/
	-  [Text](https://cs.nyu.edu/~davise/Benchmarks/Text.html)
	-   [BIG-bench](https://cs.nyu.edu/~davise/Benchmarks/BIG-bench.html) Beyond the Imitation Game collaborative benchmark for measuring and extrapolating the capabilities of language models
	- MMLU (used for chinchilla) https://towardsdatascience.com/a-new-ai-trend-chinchilla-70b-greatly-outperforms-gpt-3-175b-and-gopher-280b-408b9b4510
	-   [Images](https://cs.nyu.edu/~davise/Benchmarks/Images.html)
	-   [Videos](https://cs.nyu.edu/~davise/Benchmarks/Videos.html)
	-   [Simulated Physical Worlds](https://cs.nyu.edu/~davise/Benchmarks/Physical.html)
	-   [Symbolic/Knowledge Graphs](https://cs.nyu.edu/~davise/Benchmarks/Symbolic.html)

- riley goodside benchmark questions
	- https://scale.com/blog/chatgpt-vs-claude#Calculation
	- Overall, Claude is a serious competitor to ChatGPT, with improvements in many areas. While conceived as a demonstration of “constitutional” principles, Claude feels not only safer but more fun than ChatGPT. Claude’s writing is more verbose, but also more naturalistic. Its ability to write coherently about itself, its limitations, and its goals seem to also allow it to more naturally answer questions on other subjects.

HELM by Stanformd CRFM
- https://twitter.com/nathanbenaich/status/1610385056618663936?s=20&t=fBOWt8NvTwGGnwJ92tybAQ
- https://crfm.stanford.edu/helm/v0.2.0/?group=core_scenarios
- evaluates 34 prominent language models in a standardized way on 42 scenarios x 7 metrics.


overview of some benchmarks
https://youtu.be/EzEuylNSn-Q
- SIQA very easy to get wrong
- ascii text weird capability
- not able to do test replacement or guess hours

information retrieval benchmark
- https://github.com/beir-cellar/beir

### agent benchmarks 


- https://papersread.ai/e/agentbench-evaluating-llms-as-agents/
- autogpt benchmarking framework