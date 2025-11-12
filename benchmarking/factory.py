'''
Factory for loading models.

Gian Favero
Ideogram
2025-10-29
'''

import torch
import torch.nn as nn
from transformers import AutoModelForImageSegmentation

import sys
sys.path.insert(0, "/home/gianfavero/projects/third_party/")
sys.path.insert(0, "/home/gianfavero/projects/third_party/birefnet")
from birefnet.models.birefnet import BiRefNet
from birefnet.birefnet_utils import check_state_dict

from collections import OrderedDict

def get_model(model_name: str, device: str = "cuda", grad_enabled: bool = False, path_to_weight: str = None):
    '''
    Get a model from the factory.

    Args:
        model_name: str, the name of the model to load
        device: str, the device to load the model to
        grad_enabled: bool, whether to enable gradient computation

    Returns:
        model: nn.Module, the loaded model
    '''
    if model_name == "birefnet":
        return BiRefNet_HF(device, grad_enabled)
    elif model_name == "rmbgv2":
        return RMBGv2(device, grad_enabled)
    elif model_name == "custom":
        return BiRefNet_Custom(device, grad_enabled, path_to_weight)
    else:
        raise ValueError(f"Model {model_name} not found")

class BaseModel(nn.Module):
    '''
    Base class for all models.
    '''
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        pass

class BiRefNet_HF(BaseModel):
    '''
    BiRefNet. Hugging face code + weights.
    https://huggingface.co/ZhengPeng7/BiRefNet
    '''
    def __init__(self, device: str = "cuda", grad_enabled: bool = False):
        super(BiRefNet_HF, self).__init__()
        
        self.device = device
        
        self.model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        self.model.half()
        self.model.to(device)
        self.model.eval()

        if not grad_enabled:
            self.model.requires_grad_(False)
     
    def forward(self, x):
        return self.model(x)[-1].sigmoid()

class RMBGv2(BaseModel):
    '''
    RMBGv2. Hugging face code + weights.
    https://huggingface.co/briaai/RMBG-2.0
    '''
    def __init__(self, device: str = "cuda", grad_enabled: bool = False):
        super(RMBGv2, self).__init__()
        
        self.device = device
        
        self.model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        if not grad_enabled:
            self.model.requires_grad_(False)
     
    def forward(self, x):
        return self.model(x)[-1].sigmoid()

class BiRefNet_Custom(BaseModel):
    '''
    BiRefNet. Custom code + weights.
    '''
    def __init__(self, device: str = "cuda", grad_enabled: bool = False, path_to_weight: str = None):
        super(BiRefNet_Custom, self).__init__()

        if path_to_weight is None:
            raise ValueError("path_to_weight is required")
        
        self.device = device
        
        self.model = BiRefNet(bb_pretrained=False)
        state_dict = torch.load(path_to_weight, map_location='cpu')
        state_dict = check_state_dict(state_dict)

        # Clean up the keys
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.split(".")[1:]
            k = ".".join(k)
            k = k.replace("module._orig_mod.", "")
            clean_state_dict[k] = v

        self.model.load_state_dict(clean_state_dict)
        self.model.half()
        self.model.to(device)
        self.model.eval()

        if not grad_enabled:
            self.model.requires_grad_(False)
     
    def forward(self, x):
        return self.model(x)[-1].sigmoid()