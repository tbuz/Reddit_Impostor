import os

import torch
from data_classes.showerthought_dataset import ShowerthoughtDataset
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, GPTNeoForCausalLM, Adafactor,
                          get_linear_schedule_with_warmup  )
from transformers.optimization import AdafactorSchedule
from utils import Constants, bootstrap
from tqdm import tqdm
import wandb
import math

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 7e-6
WARMUP_STEPS = 1000
MAX_SEQ_LEN = 100

def get_lr(opt):
    lrs = [
        opt._get_lr(group, opt.state[group["params"][0]])
        for group in opt.param_groups
        if group["params"][0].grad is not None
    ]
    if len(lrs) == 0:
        lrs = LEARNING_RATE  # if called before stepping
    return lrs

bootstrap()

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=torch.float16)
model.load_state_dict(torch.load('checkpoints/gpt_neo_showerthought_3.pt'))
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id

model = model.to(Constants.device)

model.train()

proc_seq_count = 0
batch_count = 0
tmp_showerthoughts_tens = None
dataset = ShowerthoughtDataset(showerthoughts_dataset_path = '../data')
thoughts_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.995))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = len(thoughts_loader))
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=LEARNING_RATE)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(thoughts_loader))


wandb.init(project="test-project", entity="reddit-impostor", name=f"batch={BATCH_SIZE} lr={LEARNING_RATE}")
wandb.config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "warmup_steps": WARMUP_STEPS,
    "max_seq_len": MAX_SEQ_LEN
}

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    sum_loss = 0
    for batch in tqdm(thoughts_loader, total=len(thoughts_loader)):
        thoughts_tensor = tokenizer(batch[0], return_tensors="pt", padding=True, truncation=True,  max_length=MAX_SEQ_LEN)['input_ids'].clone().detach().to(Constants.device)
        for showerthought in batch[1:]:
            temp = tokenizer(showerthought, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_SEQ_LEN)['input_ids'].clone().detach().to(Constants.device)
            thoughts_tensor = torch.cat([thoughts_tensor.clone().detach(), temp], dim=1)
        
        if torch.isnan(thoughts_tensor).any().item():
            print("Input tensor contains NaN")
            continue
        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()
        optimizer.zero_grad()
        
        outputs = model(thoughts_tensor, labels=thoughts_tensor)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        # scheduler.step()
        # wandb.log({"lr": scheduler._last_lr[0]})
    
    avg_epoch_loss = sum_loss / len(thoughts_loader)
    wandb.log({"avg_epoch_loss": avg_epoch_loss})
    print(f"\n Average epoch loss: {avg_epoch_loss} \n")        
    # Store the model on every epoch
    torch.save(model.state_dict(), os.path.join(Constants.checkpoints_folder, f"gpt_neo_showerthought_{epoch + 4}.pt"))
