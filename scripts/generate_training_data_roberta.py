import os
import json
import ndjson

FILE1 = os.path.join('..',  'data', 'generated_showerthoughts_4_neo.txt')
FILE2 = os.path.join('..', 'data', 'Showerthoughts.ndjson')

def __isPostValid(post):
  if 'removed_by_category' in post and post['removed_by_category'] is not None:
      return False
  if post['selftext'] != '':
      return False
  if "post_hint" in post and post["post_hint"] == "image":
      return False
  return True

data_dir = '../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

with open(FILE1) as f:
  lines = filter(None, (line.rstrip() for line in f))
  cleanSentences = list(map(lambda x: x.replace('<|showerthought|>', '').replace('<|endoftext|>', ''), lines))[:10000]

count = 0  
with open(FILE2) as f:
    reader = ndjson.reader(f)
    try:
      for post in reader:
        if(count == 10000):
          break
        if not __isPostValid(post):
          continue
        text = post['title']
        cleanSentences.append(text)
        count += 1
    except json.JSONDecodeError:
      pass

finalData = []
temp = [*cleanSentences[:8000], *cleanSentences[10000:18000]]
# data for training
for index, sentence in enumerate(temp):
  finalData.append({'title': sentence, 
                    'label': 'generated' if index < 8000 else 'genuine'})
 # Write the data to a file
filename = f'{data_dir}/roberta_train_data.ndjson'
with open(filename, 'a') as f:
    # If we already have some data, we need to insert a newline
    ndjson.dump(finalData, f)
    
#data for testing
temp = [*cleanSentences[8000:10000], *cleanSentences[18000:]]
for index, sentence in enumerate(temp):
  finalData.append({'title': sentence, 
                    'label': 'generated' if index < 2000 else 'genuine'})
 # Write the data to a file
filename = f'{data_dir}/roberta_test_data.ndjson'
with open(filename, 'a') as f:
    # If we already have some data, we need to insert a newline
    ndjson.dump(finalData, f)