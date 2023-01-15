import itertools
import string
from collections import Counter

import gensim.downloader as api
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
import ndjson
import json

from models.data_classes.showerthought_dataset import ShowerthoughtDataset

FILE1 = "../../data/Showerthoughts.ndjson"  # original posts
FILE2 = "../../models/generated_showerthoughts_small_4.txt"  # generated posts

# Inspired by: https://intellica-ai.medium.com/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c


def pre_process(corpus):
  # convert input corpus to lower case.
  corpus = corpus.lower()
  # collecting a list of stop words from nltk and punctuation form
  # string class and create single array.
  stopset = stopwords.words('english') + list(string.punctuation)
  # remove stop words and punctuations from string.
  # word_tokenize is used to tokenize the input corpus in word tokens.
  corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
  # remove non-ascii characters
  corpus = unidecode(corpus)

  # Lematizing
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(corpus)
  for w in words:
    if (w != lemmatizer.lemmatize(w)):
        corpus = corpus.replace(w, lemmatizer.lemmatize(w))
  return corpus


def map_word_frequency(document):
    return Counter(itertools.chain(*document))


word_emb_model: KeyedVectors = api.load('word2vec-google-news-300')


def get_sif_feature_vectors(sentence1: str,
                            sentence2: str,
                            word_emb_model=word_emb_model):
  sentence1 = [
    token for token in sentence1.split()
    if token in word_emb_model.key_to_index
  ]
  sentence2 = [
    token for token in sentence2.split()
    if token in word_emb_model.key_to_index
  ]
  word_counts = map_word_frequency((sentence1 + sentence2))
  embedding_size = 300  # size of vectore in word embeddings
  a = 0.001
  sentence_set = []
  for sentence in [sentence1, sentence2]:
    vs = np.zeros(embedding_size)
    sentence_length = len(sentence)
    for word in sentence:
      a_value = a / (a + word_counts[word]
                      )  # smooth inverse frequency, SIF
      vs = np.add(vs, np.multiply(
        a_value, word_emb_model[word]))  # vs += sif * word_vector
    vs = np.divide(vs, sentence_length)  # weighted average
    sentence_set.append(vs)
  return sentence_set


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1),
                             feature_vec_2.reshape(1, -1))[0][0]


def isPostValid(post):
  if 'removed_by_category' in post:
    return False
  if post['selftext'] != '':
    return False
  if "post_hint" in post and post["post_hint"] == "image":
    return False
  return True


# sentence pair
if __name__ == "__main__":
    with open(FILE1) as f:
      reader = ndjson.reader(f)
      showerthought_list = []
      try:
        for post in reader:
          if isPostValid(post):
            showerthought_str = post['title']
            showerthought_list.append(showerthought_str)
      except json.JSONDecodeError:
        pass

    string1 = " ".join(showerthought_list)
    with open(FILE2, 'r') as f:
      lines = f.readlines()
      string2 = " ".join(lines)

    corpus = [string1, string2]
    for c in range(len(corpus)):
      corpus[c] = pre_process(corpus[c])

    temp = get_sif_feature_vectors(corpus[0], corpus[1])
    print(get_cosine_similarity(temp[0], temp[1]))
