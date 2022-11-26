# Evaluation

When our model generates posts, the question arises of what can be considered a "good" post. For this reason, we need to define criteria against which we can evaluate the posts or the model in general.

## Text Analysis

In the following, we will list different text analysis and analysis techniques we can utilise for model evaluation.

### Sentiment Analysis

One can likely assign posts of a subreddit to a similar feeling (r/jokes ➞ positive; r/news ➞ neutral), which is why this analysis is worth exploring for our use case.

### Topic Analysis

Another technique in text classification is topic analysis. Machine learning algorithms, for example, can organize texts by subject or theme.

### Intent Detection

Furthermore, it could be interesting to analyse the intent of a text. For example, it is plausible that a post in r/news is meant to be informative, while a post in r/WritingPrompts is meant to be thought-provoking.

### Keyword Extraction

Especially keywords are particularly interesting for our use case. This does not apply to every subreddit, but in r/personalfinance, for example, certain terms such as "stock" or "budget" are regularly used. We could compare whether the keywords of the subreddit match those of our generated posts.

### Word Count

This technique is straightforward. We compare the word count of our generated posts to the ones of the target subreddit.

### Word Frequency

The word frequency analysis is similar to [Keyword Extraction](#keyword-extraction): We want to identify frequent words and compare whether these words match with the posts of our target subreddit. However, instead of using machine learning like in [Keyword Extraction](#keyword-extraction), we will use an index for the word count.

### Embedding Comparison

The analysis of word embeddings is particularly interesting. We assume that the posts of a subreddit form clusters in the word vector space. This assumption still needs to be investigated, but if correct, it could become an essential factor in our evaluation.

## Human Evaluation

Finally, we came up with two ideas to have people evaluate the results. Although this can only be done for a small number of posts, it is crucial also to assess how a human would evaluate the generated posts.

### Post in Subreddit

As soon as our models generate the first posts, we could post them in the associated subreddit and investigate how the subreddit communities react to them. Among other things, the posts might be deleted. In this case, it would be worth finding out the reason for deletion, as it could, for example, be classified as unsuitable in the context of the subreddit, or maybe it was too obvious that it is machine-generated.

### Turing Test

Furthermore, a [Turing Test](https://en.wikipedia.org/wiki/Turing_test) would be interesting, which could be carried out with few persons. For example, 5-10 actual posts and 5-10 generated posts could be presented in random order. The task of the test persons is to classify these as generated or real. A low level of distinguishability indicates a good model.
