import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, TrainingArguments, Trainer
from data_classes.showerthought_dataset import ShowerthoughtDataset

EPOCHS=5
BATCH_SIZE=8
WARMUP_STEPS=5000
MAX_LENGTH = 70

default_args = {
    "output_dir": "tmp",
    "num_train_epochs": EPOCHS,
    "log_level": "error"
}


model_name = "EleutherAI/gpt-neo-2.7B"
torch.manual_seed(42)

tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|showerthought|>', eos_token='<|endoftext|>')
model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model.resize_token_embeddings(len(tokenizer))

dataset = ShowerthoughtDataset(showerthoughts_dataset_path = '../data')

training_args = TrainingArguments(logging_steps=100,
                                  save_strategy="epoch", per_device_train_batch_size=1,gradient_accumulation_steps=4,
                                  gradient_checkpointing=True, per_device_eval_batch_size=1, weight_decay=0.01,
                                  warmup_steps=WARMUP_STEPS,
                                  optim="adafactor", **default_args)

Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=lambda data: {
  'input_ids': torch.cat([torch.tensor(tokenizer(d, truncation=True, max_length=MAX_LENGTH)['input_ids']) for d in data]),
  'attention_mask': torch.cat([torch.tensor(tokenizer(d, truncation=True, max_length=MAX_LENGTH)['attention_mask']) for d in data]),
  'labels': torch.cat([torch.tensor(tokenizer(d, truncation=True, max_length=MAX_LENGTH)['input_ids']) for d in data]),
}).train()
