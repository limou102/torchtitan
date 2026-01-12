import logging

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

def print_tensor(t : torch.Tensor, name : str):
    print("<------------{}".format(name))
    if torch.is_tensor(t):
        print("shape={}, dtype={}, device={}, contiguous={}, requires_grad={}".format(
            t.shape, t.dtype, t.device, t.is_contiguous(), t.requires_grad))
        if isinstance(t, DTensor):
            print("DTensor Found")
        flat = t.flatten() 
        print("value : {} ... {}".format(flat[:10], flat[-10:]))
    else:
        print("NOT A TENSOR, type={}, value={}".format(type(t), t))
    print("{}------------>\n".format(name), flush=True)