import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import Constants

MODEL_EPOCH = input('Enter which EPOCH you would like to use: ')

models_folder = Constants.checkpoints_folder

# Comment out this line if you want to use a GPU
map_location=torch.device('cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model_path = os.path.join(models_folder, f"gpt2_medium_showerthought_3.pt")
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.load_state_dict(torch.load(model_path, map_location=map_location))

showerthoughts_output_file_path = f'generated_showerthoughts_{MODEL_EPOCH}.txt'

model.eval()
if os.path.exists(showerthoughts_output_file_path):
    os.remove(showerthoughts_output_file_path)
    
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


thoughts_num = 0
with torch.no_grad():
    
    for thought_idx in range(100):
        thought_finished = False
        cur_ids = torch.tensor(tokenizer.encode("<|showerthought|>")).unsqueeze(0).to(Constants.device)
        
        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(Constants.device) * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                thought_finished = True
                break
        
        if thought_finished:
            thoughts_num = thoughts_num + 1
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)
            with open(showerthoughts_output_file_path, 'a') as f:
                f.write(f"{output_text} \n\n")
            
