import os

import torch
from data_classes.showerthought_dataset import ShowerthoughtDataset
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer, GPTJForCausalLM,
                          get_linear_schedule_with_warmup)
from utils import Constants, bootstrap

# Hyperparameters
BATCH_SIZE = 12
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

bootstrap()

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')

model.parallelize({
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        1: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    })
model = model.to(Constants.device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_showerthoughts_tens = None
dataset = ShowerthoughtDataset(showerthoughts_dataset_path = '../data')
thoughts_loader = DataLoader(dataset, batch_size=1, shuffle=True)

with torch.cuda.amp.autocast():
    
    for epoch in range(EPOCHS):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx,showerthought in enumerate(thoughts_loader):
            
            #################### "Fit as many showerthought sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            showerthought_tens = torch.tensor(tokenizer.encode(showerthought[0])).unsqueeze(0).to(Constants.device)
            #Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if showerthought_tens.size()[1] > MAX_SEQ_LEN:
                continue
            
            #The first showerthought sequence in the sequence
            if not torch.is_tensor(tmp_showerthoughts_tens):
                tmp_showerthoughts_tens = showerthought_tens
                continue
            else:
                #The next showerthought does not fit in so we process the sequence and leave the last showerthought 
                #as the start for next sequence 
                if tmp_showerthoughts_tens.size()[1] + showerthought_tens.size()[1] > MAX_SEQ_LEN:
                    work_showerthoughts_tens = tmp_showerthoughts_tens
                    tmp_showerthoughts_tens = showerthought_tens
                else:
                    #Add the showerthought to sequence, continue and try to add more
                    tmp_showerthoughts_tens = torch.cat([tmp_showerthoughts_tens, showerthought_tens[:,1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################
                
            outputs = model(work_showerthoughts_tens, labels=work_showerthoughts_tens)
            loss, logits = outputs[:2]                        
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
            
            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0    
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0
        
        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(Constants.checkpoints_folder, f"gptj_showerthought_{epoch}.pt"))
