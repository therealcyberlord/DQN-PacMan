import torch 

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")