import torch
import os

class Constants:
    checkpoints_folder = "checkpoints"
    device = 'cpu'
    
def bootstrap():
    if torch.cuda.is_available():
        Constants.device = 'cuda'

    print('Device is: ', Constants.device)
    
    if not os.path.exists(Constants.checkpoints_folder):
        os.mkdir(Constants.checkpoints_folder)
    
