# Word Embeddings

In Natural Language Processing (NLP), the term "Word Embedding" refers to the concept of representing the words of a corpus as vectors.

## Bag of Words

Bag-of-Words (BoF) is a simple NLP technique to extract features from a given input text. The BoF keeps track of the number of occurrences of words while discarding information about the order, structure, and document are discarded.

For example, suppose that the following two sentences are given.

- **Sentence 1:** This is an example
- **Sentence 2:** Having an example is great.

Our corpus would contain the following words: `["This", "is", "an", "example", "Having", "great"]`.

The BoF would map the first sentence to the vector `[1,1,1,1,0,0]` and the second sentence to `[0,1,1,1,0,1,1]`.

Of course, this is the most simple version of BoF and could be extended by preprocessing. The preprocessing could, for example, include the removal of Stop-Words, removing special characters, or normalizing the words.

## Word2Vec

Word2Vec uses a neural network, which takes a large corpus of words as input and produces a multi-dimensional vector space as an output, in which each word is assigned a unique vector. The vectors commonly consist of floating-point numbers such that words with similar meanings are close in vector space.

This means that Word2Vec, in-contrary to BoW, store contextual information about the words. This representation not only allows finding similar words but also allows doing arithmetic operations on these words. If one were to calculate "king" - "man" + "woman", the closest vector in a well-trained model would be the one representing the word "queen".

## Sources

- https://en.wikipedia.org/wiki/Word_embedding
- https://de.wikipedia.org/wiki/Worteinbettung
- https://www.youtube.com/watch?v=gQddtTdmG_8
- https://israelg99.github.io/2017-03-23-Word2Vec-Explained/
