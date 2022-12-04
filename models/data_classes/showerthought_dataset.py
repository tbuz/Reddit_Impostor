from torch.utils.data import Dataset
import os
import json
import ndjson


class ShowerthoughtDataset(Dataset):
    def __init__(self, showerthoughts_dataset_path = '../../data'):
        super().__init__()

        short_showerthoughts_path = os.path.join(showerthoughts_dataset_path, 'Showerthoughts.ndjson')

        self.showerthought_list = []
        self.begin_of_text_token = "<|showerthought|>"
        self.end_of_text_token = "<|endoftext|>"

        with open(short_showerthoughts_path) as f:
          reader = ndjson.reader(f)
          try:
            for post in reader:
              if self.__isPostValid(post):
                showerthought_str = f"{self.begin_of_text_token}{post['title']}{self.end_of_text_token}"
                self.showerthought_list.append(showerthought_str)
          except json.JSONDecodeError:
            pass
          
    def __isPostValid(self, post):
      if 'removed_by_category' in post:
          return False
      if post['selftext'] != '':
          return False
      if "post_hint" in post and post["post_hint"] == "image":
          return False
      return True
      
    def __len__(self):
        return len(self.showerthought_list)

    def __getitem__(self, item):
        return self.showerthought_list[item]