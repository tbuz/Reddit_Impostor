from transformers import RobertaForSequenceClassification, AutoTokenizer
from utils import Constants, bootstrap
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_classes.roberta_dataset import RobertaDataset
import os
import numpy as np
from sklearn.metrics import classification_report
import ndjson
import json

#Hyperparameters
BATCH_SIZE = 1
MAX_SEQ_LEN = 512

bootstrap()
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.load_state_dict(torch.load('checkpoints/roberta_32.pt'))
model = model.to(Constants.device)

model.eval()

dataset = RobertaDataset(tokenizer=tokenizer,max_token_len=MAX_SEQ_LEN, showerthoughts_dataset_path='../data', isValidation=True)
          
thoughts_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

predictions, true_labels = [], []
for batch in tqdm(thoughts_loader, total=len(thoughts_loader)):

  model.zero_grad()
  
  outputs = model(batch['input_ids'].to(Constants.device), attention_mask=batch['attention_mask'].to(Constants.device))
  logits = outputs.logits.detach().to('cpu')
  labels = batch['labels'].to('cpu')
  prediction = 'genuine' if np.argmax(logits.numpy()).flatten().item() == 1.0 else 'generated'
  true_label = 'genuine' if labels.flatten().item() == 1 else 'generated'
  predictions.append(prediction)
  true_labels.append(true_label)

cr = classification_report(true_labels, predictions, output_dict=False)
print(cr)




sentences = []
path = '../data/roberta_test_data.ndjson'
with open(path) as f:
  reader = ndjson.reader(f)
  try:
    for post in reader:
        sentences.append(post)
  except json.JSONDecodeError:
    pass
sentences1 = []
sentences2 = []
for idx in tqdm(range(len(predictions))):
  if predictions[idx] == 'genuine' and true_labels[idx] == 'generated':
    sentences1.append({"title": sentences[idx]['title']})   
  if predictions[idx] == 'generated' and true_labels[idx] == 'genuine':
    sentences2.append({"title": sentences[idx]['title']})   
 
      
with open(f'genuine_but_was_generated.ndjson', 'a') as f1:
  ndjson.dump(sentences1, f1)
with open(f'generated_but_was_genuine.ndjson', 'a') as f2:
  ndjson.dump(sentences2, f2)
    
 