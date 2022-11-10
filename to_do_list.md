# To Do List

## 1. Preparation & Research

- ~~Research: Which subreddits are interesting as text sources? (differences regarding length, fictional/non-fictional / facts)~~
  - ~~r/todayilearned~~
  - ~~r/Showerthoughts~~
  - ~~r/Jokes~~
  - ~~r/LifeProTips~~
  - ~~r/AskReddit~~
  - ~~r/AskScience~~
  - ~~r/stories~~
  - ~~r/news, r/worldnews, r/politics~~
  - r/books
  - r/explainlikeimfive
  - r/confession
  - r/WritingPrompts
  - r/personalfinance
  - r/confession
  - r/tifu
  - ~~=> Can you compare these in a table? What are important criteria to differentiate them?~~
- ~~Research: What are Word Embeddings? (Bag of Words, CountVectorizer, Word2Vec)~~
- ~~Research: Which language models are relevant / interesting?~~
  - ~~Markov Chain~~
  - ~~RNN/LSTM~~
  - ~~GPT-(2/3/neo)~~
  - ~~BERT / Transformers~~
  - ~~T5~~
  - ~~XLNet~~
- Research: Related research on social media bots, text style transfer, Reddit communities, ...
  - Use sources like: Google Scholar, Towardsdatascience, etc.
- Research & brainstorm: How to evaluate the success of a generated post?
  - How do other researchers measure similar tasks?
  - Kind of a "Turing Test" - Survey with 10 real and 10 fake posts
  - Post best posts on subreddit
  - ...
- Research Embeddings:
  - What is cosine similarity?
  - spacy for embeddings, NER, sentiment analysis, etc. (powerful package) https://spacy.io/
  - SBERT (pre-trained)
  - Is VADER sentiment analysis helpful?

## 2. Dataset Acquisition & Exploration

- Collecting a sufficiently large dataset from selected subreddits
  - focus on subreddits with text (avoid news etc for now)
- ~~Look at Reddit API, Pushshift API~~
- ~~Pushshift API => Build a script to access the API~~
- ~~Compare different subreddits (avg word length, how much facts vs. story?, links or selftext, etc) => create table~~
- ...

## 3. Model Exploration / Training / Fine-Tuning & Testing

- https://huggingface.co/models for model options
- Implement pre-trained models
- How to fine-tune a Language Model like BERT?
- ...

## 4. Evaluation

- Descriptive / statistical evaluation of generated texts (word count, vocabulary, etc.) - research on how to quantify/compare/evaluate text data
- How is the grammar, etc.
- Post it in the subreddit => how does the community react?
- Unsupervised approaches => compare embeddings
- Sentiment analysis
- LIWC as additional benchmark / comparison?

## 5. Additional ideas

- Can we build a model that summarizes top 5 news of the day?
- ...
