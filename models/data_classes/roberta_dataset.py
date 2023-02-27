from torch.utils.data import Dataset
import os
import json
import ndjson
import torch
from transformers import PreTrainedTokenizer


class RobertaDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_token_len: int, showerthoughts_dataset_path = '../../data', isValidation=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        short_showerthoughts_path = os.path.join(showerthoughts_dataset_path, 'roberta_train_data.ndjson' if not isValidation else 'roberta_test_data.ndjson')

        self.showerthought_list = []

        with open(short_showerthoughts_path) as f:
          reader = ndjson.reader(f)
          try:
            for post in reader:
                self.showerthought_list.append(post)
          except json.JSONDecodeError:
            pass
      
    def __len__(self):
        return len(self.showerthought_list)

    def __getitem__(self, item):
      item = self.showerthought_list[item]
      title = str(item['title'])
      labels = torch.tensor([1 if item['label'] == 'genuine' else 0])
      tokens = self.tokenizer.encode_plus(title, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=self.max_token_len,
                                          padding='max_length', return_attention_mask=True)
      return {'input_ids': tokens['input_ids'].flatten(), 'attention_mask': tokens['attention_mask'].flatten(), 'labels': labels}