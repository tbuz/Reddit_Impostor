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
import plotly.express as px
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import random


FILE1 = os.path.join('..', '..', 'data', 'Showerthoughts-small.ndjson') # PATH to file 1
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
  cleanSentences1 = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))[:5000]

with open(FILE3) as f:
  lines = filter(None, (line.rstrip() for line in f))
  cleanSentences2 = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))[:5000]

showerthought_list = random.sample(showerthought_list, 1000)
cleanSentences1 = random.sample(cleanSentences1, 1000)
cleanSentences2 = random.sample(cleanSentences2, 1000)

np.random.seed(100)
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sentences are encoded by calling model.encode()
print('Encoding embeddings...')
embeddings = model.encode([*showerthought_list, *cleanSentences1, *cleanSentences2])

print('Reducing dimensions with TSNE...')
n_components = 2
tsne = TSNE(n_components=n_components)
tsne_result = tsne.fit_transform(embeddings)

print('Clustering with OPTICS...')
optics_clustering = OPTICS(min_samples=5, min_cluster_size=0.005).fit(tsne_result)

print('Creating DataFrame')
df = pd.DataFrame()
df['x'] = tsne_result[:,0]
df['y'] = tsne_result[:,1]
df['clustering'] = optics_clustering.labels_
df['datasets'] = [*(['Genuine'] * len(showerthought_list)), *(['GPT Neo'] * len(cleanSentences1)), *(['GPT2'] * len(cleanSentences2))]
df['sentences'] = [*showerthought_list, *cleanSentences1, *cleanSentences2]


print('Removing unclustered data...')
unclustered = {
  'genuine': 0,
  'gptNeo': 0,
  'gpt2':0,
}
for index, row in df.iterrows():
  if row['clustering'] == -1:
    if row['datasets'] == 'Genuine':
      unclustered['genuine'] += 1
    if row['datasets'] == 'GPT Neo':
      unclustered['gptNeo'] += 1
    if row['datasets'] == 'GPT2':
      unclustered['gpt2'] += 1
    df.drop(index, inplace=True)



clustersDict = {cluster: [] for cluster in optics_clustering.labels_}
for index, row in df.iterrows():
  clustersDict[row['clustering']].append(row['sentences'])

# remove unclustered
del clustersDict[-1]

# Add a topic column and set the topics initially to None
df['topic'] = None

# print('Adding topics...')
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
# topicModel = BERTopic(embedding_model=model) # use the same model that we use for creating the embeddings
# topicModel.fit_transform(docs)
# topicModel.generate_topic_labels()
# # key is the cluseter and value is the array of sentences in that cluster
# for key in clustersDict:  
#   topics, probs = topicModel.transform((clustersDict[key]))
#   topicIndex = topics[np.argmax(probs)] # take the index of the most probable topic
#   topic = topicModel.topic_labels_[topicIndex]
#   df.loc[df['clustering'] == key, 'topic'] = topic
    
# fig = px.scatter(
#   df, x='x', y='y',
#   color='datasets', labels={'color': 'datasets'},
#   hover_data=['sentences', 'topic']
# )
# fig.show()
# fig.write_html('plot.html')

x = sns.scatterplot(x='x', y='y', data=df, s=5, hue='datasets', palette=sns.color_palette('hls',3), legend=True)
x.set(title="Showerthoughts similarity")
x.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
x.tick_params(bottom=False, left=False)
plt.savefig('plot.svg', transparent=True)

print(f"Identified clusters: {optics_clustering.labels_.max() + 1}")
print(unclustered)




    