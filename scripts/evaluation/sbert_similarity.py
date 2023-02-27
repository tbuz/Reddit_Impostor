from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ndjson
import json
import os


FILE1 = os.path.join('..', '..', 'data', 'Showerthoughts-small.ndjson') # PATH to file 1
FILE2 = os.path.join('..', '..', 'data', 'generated_showerthoughts_3.txt')

def __isPostValid(post):
  if 'removed_by_category' in post:
      return False
  if post['selftext'] != '':
      return False
  if "post_hint" in post and post["post_hint"] == "image":
      return False
  return True

showerthought_list = []

with open(FILE1) as f:
  reader = ndjson.reader(f)
  try:
    for post in reader:
      if __isPostValid(post):
        showerthought_str = post['title']
        showerthought_list.append(showerthought_str)
  except json.JSONDecodeError:
    pass

with open(FILE2) as f:
  lines = filter(None, (line.rstrip() for line in f))
  cleanSentences = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))

np.random.seed(100)
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sentences are encoded by calling model.encode()
embeddings = model.encode([*showerthought_list[:100], *cleanSentences])

n_components = 2
tsne = TSNE(n_components=n_components)
tsne_result = tsne.fit_transform(embeddings)

optics_clustering = OPTICS(min_samples=5, min_cluster_size=0.005).fit(tsne_result)

df = pd.DataFrame()
df['x'] = tsne_result[:,0]
df['y'] = tsne_result[:,1]
df['clustering'] = optics_clustering.labels_
df['datasets'] = np.concatenate((np.zeros(100, dtype=int), np.ones(100, dtype=int)))

amountOfUnclusteredElements = 0
for index, row in df.iterrows():
  if row['clustering'] == -1:
    amountOfUnclusteredElements += 1
    df.drop(index, inplace=True)

sns.scatterplot3d(x='x', y='y',z='datasets', data=df, hue='datasets', palette=sns.color_palette('hls', 2), legend=True)
plt.savefig('plot.svg')

print(f"Identified clusters: {optics_clustering.labels_.max() + 1}")
print(f"Amount of sentences that weren't clustered: {amountOfUnclusteredElements} out of {embeddings.shape[0]}")



    