import json
import os

import matplotlib.pyplot as plt
import ndjson
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

filenames = ['Showerthoughts-small', 'books', 'personalfinance', 'generated_showerthoughts_4_neo', 'confession', 'politics', 'askscience']

def __isPostValid(post):
  if 'removed_by_category' in post and post['removed_by_category'] is not None:
      return False
  # if post['selftext'] != '':
  #     return False
  if "post_hint" in post and post["post_hint"] == "image":
      return False
  return True

items = []

for filename in filenames:
  path = os.path.join('..', '..', 'data', filename + '.ndjson')
  count = 0
  with open(path) as f:
    reader = ndjson.reader(f)
    try:
      for post in reader:
        if(count == 1000):
          break
        if(filename != 'generated_showerthoughts_4_neo'):
          if not __isPostValid(post):
            continue
        text = post['title']
        if 'selftext' in post:
          text += ' ' + post['selftext']
        temp = {
          'subreddit': filename,
          'text': text,
        }
        items.append(temp)
        count += 1
    except json.JSONDecodeError:
      pass

np.random.seed(100)
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sentences are encoded by calling model.encode()
print('Encoding embeddings...')
embeddings = model.encode([item['text'] for item in items])

print('Reducing dimensions with TSNE...')
n_components = 2
tsne = TSNE(n_components=n_components)
tsne_result = tsne.fit_transform(embeddings)

# add tsne result to items
for i in range(len(items)):
  items[i]['x'] = tsne_result[i][0]
  items[i]['y'] = tsne_result[i][1]

print('Creating DataFrame')
df = pd.DataFrame(items)
    
# fig = px.scatter(
#   df, x='x', y='y',
#   color='subreddit',
#   hover_data=['text']
# )
# fig.show()

x = sns.scatterplot(x='x', y='y',s=5, data=df, hue='subreddit',
                    palette=sns.color_palette('hls', len(filenames)),
                    legend=True)
x.set(title='Subreddits embeddings')
x.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
x.tick_params(bottom=False, left=False)
plt.savefig('plot.svg', transparent=True)
