from transformers import GPTNeoForCausalLM, AutoTokenizer
from utils import Constants, bootstrap
import torch
from tqdm import tqdm
import os

MODELNAME = "gpt_neo_showerthought_4.pt"
showerthoughts_output_file_path = os.path.join('..', 'data', f'generated_showerthoughts_4_neo.txt')

model_folder = Constants.checkpoints_folder

bootstrap()

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=torch.float16)
model.load_state_dict(torch.load('checkpoints/gpt_neo_showerthought_4.pt'))
model.resize_token_embeddings(len(tokenizer))

model = model.to(Constants.device)
model.eval()
with torch.no_grad():
  with open(showerthoughts_output_file_path, 'a') as f:
    for i in tqdm(range(5000)):
      promptTensor = tokenizer(f"<|showerthought|>", return_tensors="pt")['input_ids'].clone().detach().to(Constants.device)
      output = model.generate(promptTensor,top_k=15, do_sample=True,max_length=100, temperature=0.9)
      text = tokenizer.decode(output[0], skip_special_tokens=False)
      f.write(f"{text}\n")
    