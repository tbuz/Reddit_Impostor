from textstat import textstat
import os
import json
import ndjson
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import language_tool_python
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')


FILE1 = os.path.join('..', '..', 'data', 'Showerthoughts.ndjson') # PATH to file 1
FILE2 = os.path.join('..', '..', 'data', 'generated_showerthoughts_4_neo.txt')
FILE3 = os.path.join('..', '..', 'data', 'generated_showerthoughts_4_gpt2.txt')


def __isPostValid(post):
  if 'removed_by_category' in post:
      return False
  if post['selftext'] != '':
      return False
  if "post_hint" in post and post["post_hint"] == "image":
      return False
  return True

def vocabularySize(sentences: list):
  vocab = set()
  for sentence in sentences:
    words = sentence.split()
    vocab.update(words)
  return len(vocab)

# Lemmatizing a string
def lematize(corpus: str):
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(corpus)
  for w in words:
    if (w != lemmatizer.lemmatize(w)):
      corpus = corpus.replace(w, lemmatizer.lemmatize(w))
  return corpus
    
  

showerthought_list = []

with open(FILE1) as f:
  reader = ndjson.reader(f)
  try:
    for post in reader:
      if(len(showerthought_list) == 5000):
        break
      if __isPostValid(post):
        showerthought_str = post['title']
        showerthought_list.append(showerthought_str)
  except json.JSONDecodeError:
    pass

with open(FILE2) as f:
  lines = filter(None, (line.rstrip() for line in f))
  cleanSentences1 = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))


with open(FILE3) as f:
  lines = filter(None, (line.rstrip() for line in f))
  cleanSentences2 = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))

with language_tool_python.LanguageTool('en-US') as tool:
  results1 = [] #Array to calucalate the variance from
  lengthArray1 = []
  grammarlMistakes1 = []
  for item in tqdm(showerthought_list[:5000]):
    results1.append(textstat.text_standard(item, float_output=True))
    lengthArray1.append(len(item))
    grammarlMistakes1.append(len(tool.check(item)))

  results2 = []
  lengthArray2 = []
  grammarlMistakes2 = []
  for item in tqdm(cleanSentences1):
    results2.append(textstat.text_standard(item, float_output=True))
    lengthArray2.append(len(item))
    grammarlMistakes2.append(len(tool.check(item)))


  results3 = []
  lengthArray3 = []
  grammarlMistakes3 = []
  for item in tqdm(cleanSentences2):
    results3.append(textstat.text_standard(item, float_output=True))
    lengthArray3.append(len(item))
    grammarlMistakes3.append(len(tool.check(item)))

difficultWordsFile1 = textstat.difficult_words(lematize(''.join(showerthought_list)))
difficultWordsFile2 = textstat.difficult_words(lematize(''.join(cleanSentences1)))
difficultWordsFile3 = textstat.difficult_words(lematize(''.join(cleanSentences2)))

### MAKING TABLE
x = PrettyTable()
x.field_names = ['File','Average complexity', 'Standard diviation (complexity)', 'Average length', 'Standard deviation (legth)' ,'Vocab size', 'Dif words/sentence', 'Grammar mistakes/sentence']
x.add_row(['Original', np.average(results1), np.std(results1), np.average(lengthArray1), np.std(lengthArray1), vocabularySize(showerthought_list), difficultWordsFile1/len(showerthought_list), np.average(grammarlMistakes1) ])
x.add_row(['GPTNeo', np.average(results2), np.std(results2), np.average(lengthArray2), np.std(lengthArray2), vocabularySize(cleanSentences1), difficultWordsFile2/len(cleanSentences1), np.average(grammarlMistakes2)])
x.add_row(['GPT2', np.average(results3), np.std(results3), np.average(lengthArray3), np.std(lengthArray3),vocabularySize(cleanSentences2), difficultWordsFile3/len(cleanSentences2), np.average(grammarlMistakes3)])

print(x)

