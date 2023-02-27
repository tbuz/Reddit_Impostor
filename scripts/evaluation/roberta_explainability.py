from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import torch
import ndjson
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.load_state_dict(torch.load('../../models/checkpoints/roberta_16.pt'))
model.to('cuda')
model.eval()

# With both the model and tokenizer initialized we are now able to get explanations on an example text.
data = []
path = '../../data/roberta_train_data.ndjson'
with open(path) as f:
  reader = ndjson.reader(f)
  try:
    for post in reader:
        data.append(post)
  except json.JSONDecodeError:
    pass

cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)

realCounter = {}
fakeCounter = {}

for post in tqdm(data):
  word_attributions = cls_explainer(post['title'])
  word_attributions.sort(key=lambda x: x[1], reverse=True)
  top4 = word_attributions[:4]
  if cls_explainer.predicted_class_index == 0: # generated class has an index of 0
    for entry in top4:
      if not entry[0] in fakeCounter:
        fakeCounter[entry[0]] = []
      fakeCounter[entry[0]].append(entry[1])
      for key in fakeCounter:
        if not key == entry[0]:
          fakeCounter[key].append(0)
  else: 
    for entry in top4:
      if not entry[0] in realCounter:
        realCounter[entry[0]] = []
      realCounter[entry[0]].append(entry[1])
      for key in realCounter:
        if not key == entry[0]:
          realCounter[key].append(0)
      
for key in fakeCounter:
  fakeCounter[key] = np.average(fakeCounter[key])
for key in realCounter:
  realCounter[key] = np.average(realCounter[key])

#sort them
fakeCounter = sorted(fakeCounter.items(), key=lambda item: item[1], reverse=True)
realCounter = sorted(realCounter.items(), key=lambda item: item[1], reverse=True)


fakeCounterDf = pd.DataFrame(fakeCounter[:10])
# 0 - the keys of the dict, 1 - the values
x = sns.barplot(fakeCounterDf, x=0, y=1)
x.set(title="Top contributers to generated class")
plt.savefig('generated_contributers.svg')

realCounterDf = pd.DataFrame(realCounter[:10])
# 0 - the keys of the dict, 1 - the values
x = sns.barplot(realCounterDf, x=0, y=1)
x.set(title="Top contributers to genuine class")
plt.savefig('genuine_contributers.svg')

