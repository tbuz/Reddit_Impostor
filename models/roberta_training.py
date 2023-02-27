from transformers import RobertaForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from utils import Constants, bootstrap
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_classes.roberta_dataset import RobertaDataset
import math
import wandb
import os

#Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
MAX_SEQ_LEN = 512

bootstrap()
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model = model.to(Constants.device)

model.train()

dataset = RobertaDataset(tokenizer=tokenizer,max_token_len=MAX_SEQ_LEN, showerthoughts_dataset_path='../data')
          
thoughts_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=(len(thoughts_loader) * EPOCHS))

wandb.init(project="roberta", entity="reddit-impostor", name=f"batch={BATCH_SIZE}")
wandb.config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "max_seq_len": MAX_SEQ_LEN
}

for epoch in range(EPOCHS):
  print(f"EPOCH {epoch} started" + '=' * 30)
  sum_loss = 0
  for batch in tqdm(thoughts_loader, total=len(thoughts_loader)):
 
    model.zero_grad()
    optimizer.zero_grad()
    
    outputs = model(batch['input_ids'].to(Constants.device), attention_mask=batch['attention_mask'].to(Constants.device), labels=batch['labels'].to(Constants.device))
    loss = outputs.loss
    logits = outputs.logits
    sum_loss = sum_loss + loss.detach().item()
    loss.backward()
    wandb.log({"loss": loss})
    if (math.isnan(loss)):
      print("Loss became NaN")
      exit(1)
    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
    optimizer.step()
    scheduler.step()
  
  
  avg_epoch_loss = sum_loss / len(thoughts_loader)
  wandb.log({"avg_epoch_loss": avg_epoch_loss})
  print(f"\n Average epoch loss: {avg_epoch_loss} \n")

torch.save(model.state_dict(), os.path.join(Constants.checkpoints_folder, f"roberta_{BATCH_SIZE}.pt"))
 