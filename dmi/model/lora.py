import torch
import torch.nn as nn
from dmi.model.projector import Projector
from dmi.utils.args import LoraArgs, setup_args

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x


class LoraAdapters(nn.Module):
    def __init__(self, lora_args: LoraArgs, lm_emb_dim, mm_emb_dim, device):
        super().__init__()
        self.lm_emb_dim = lm_emb_dim
        self.mm_emb_dim = mm_emb_dim
        self.device = device
        setup_args(self, prefix='lora_', args=lora_args)
        self.build_model()

    def build_model(self):
        loras = []
        for layer_idx in range(self.n_proj_layers):
            adapter = LoRALayer(self.mm_emb_dim if layer_idx == 0 else self.lm_emb_dim, self.lm_emb_dim, self.rank, self.alpha)
            loras.append(adapter)
        
        self.loras = nn.ModuleList(loras)

    def forward(self, x):
        pass


class LoraWrapper(nn.Module):
    def __init__(self, lora_args, proj_args, lm_emb_dim, mm_emb_dim, device):
        super(LoraWrapper, self).__init__()
        self.device = device
        self.lora_adapters = LoraAdapters(lora_args, lm_emb_dim, mm_emb_dim, device)
        self.projector = Projector(proj_args, lm_emb_dim, mm_emb_dim, device)
        self.projector.load_model()

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.projector.eval()
        return self
    
    def forward(self, x):
        return self.projector.only_lora_forward(x, self.lora_adapters.loras)

    def trainable_parameters(self):
        return self.lora_adapters.parameters()
    