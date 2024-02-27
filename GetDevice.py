import torch

def getDevice():
    # check for cuda
    if torch.cuda.is_available():
        print("cuda is available – using gpu")
        device = torch.device("cuda")  # default to the first cuda device
    elif torch.backends.mps.is_available():
        print("mps is available! Using Apple Silicon gpu")
        device = torch.device("mps")
    else:
        print("using cpu,")
        device = torch.device("cpu")

    return device