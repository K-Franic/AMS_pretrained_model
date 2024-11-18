import torch
def device(tensor):
    return (torch.device("cuda" if torch.cuda.is_available() else "cpu"))