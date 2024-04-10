import torch

cpu: torch.device = torch.device("cpu")
device: None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")