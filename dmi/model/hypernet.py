import math
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from dmi.model.projector import Projector
from dmi.utils.args import HypnetArgs, setup_args


#################################################################################################
# This function and the next class incorporate code from RobertCsordas/transformer_generalization
# Licenced under the MIT Licence: https://mit-license.org/
#################################################################################################
def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
                             device: Optional[torch.device] = None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=128, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        scale = 1.0 / math.sqrt(d_model)
        pe = sinusoidal_pos_embedding(d_model, max_len, 0, device) * scale

        self.batch_dim = 0
        pe = pe.unsqueeze(self.batch_dim)

        self.register_buffer('pe', pe)

    def get(self, n: int, offset: int) -> torch.Tensor:
        return self.pe.narrow(1 - self.batch_dim, start=offset, length=n)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.get(x.size(1 - self.batch_dim), offset)
        return self.dropout(x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None): # [n, seq_len, d_model]
        q = self.q(x)# [n, seq_len, d_model]
        k = self.k(x)# [n, seq_len, d_model]
        v = self.v(x)# [n, seq_len, d_model

        q = q.view(*q.shape[:2], self.nhead, self.d_model // self.nhead)# [n, seq_len, heads, d_model // heads]
        k = k.view(*k.shape[:2], self.nhead, self.d_model // self.nhead)# [n, seq_len, heads, d_model // heads]
        v = v.view(*v.shape[:2], self.nhead, self.d_model // self.nhead)# [n, seq_len, heads, d_model // heads]

        q = q.transpose(1, 2)# [n, heads, seq_len, d_model // heads]
        k = k.transpose(1, 2)# [n, heads, seq_len, d_model // heads]
        v = v.transpose(1, 2)# [n, heads, seq_len, d_model // heads]

        # [n, heads, seq_len, d_model // heads] @ [n, heads, d_model // heads, seq_len] --> [n, heads, seq_len, seq_len]
        scores = q @ k.transpose(-2, -1) / np.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1) # [n, heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        attention = attn_weights @ v # [n, heads, seq_len, d_model // heads]
        attention = attention.transpose(1, 2) # [n, seq_len, heads, d_model // heads]
        attention = attention.contiguous().view(*attention.shape[:2], self.d_model)
        
        return attention
    
class HyperNetwork(nn.Module):
    def __init__(
            self, hn_args: HypnetArgs, lm_emb_dim, mm_emb_dim, n_tokens, device
        ):
        super().__init__()
        setup_args(self, prefix='hn_', args=hn_args)

        self.lm_emb_dim = lm_emb_dim
        self.mm_emb_dim = mm_emb_dim
        self.n_tokens = n_tokens
        self.device = device

        if self.arch == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hypnet_dim, dim_feedforward=self.hypnet_dim*4, nhead=self.n_heads, batch_first=True, activation='gelu')
            self.hypnet = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        elif self.arch == 'attention':
            self.hypnet = MultiheadSelfAttention(d_model=self.hypnet_dim, nhead=self.n_heads)
        elif self.arch == 'att_w_nonlinear':
            self.hypnet = nn.Sequential(
                MultiheadSelfAttention(d_model=self.hypnet_dim, nhead=self.n_heads),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown hypernetwork architecture: {self.arch}")

        generators, a_dims, b_dims = [], [], []
        
        for layer_idx in range(self.n_proj_layers):
            if layer_idx == 0:
                a_dim, b_dim = (self.hypnet_dim * self.rank), (self.rank * self.lm_emb_dim)
            else:
                a_dim, b_dim = (self.lm_emb_dim * self.rank), (self.rank * self.lm_emb_dim)

            weight_dim = a_dim + b_dim
            if self.predict_bias:
                weight_dim += self.lm_emb_dim

            generator = nn.Linear(self.hypnet_dim, weight_dim)
            generators.append(generator)
            a_dims.append(a_dim)
            b_dims.append(b_dim)
        
        self.generators = nn.ModuleList(generators)
        self.a_dims = a_dims
        self.b_dims = b_dims

        self.prefix_tokens = nn.Parameter(torch.randn((self.n_proj_layers, self.hypnet_dim)))

        if self.use_pos_encs:
            # +1 for the prefix token
            self.pos_encs = PositionalEncoding(self.hypnet_dim, max_len=2*n_tokens+len(self.prefix_tokens)+1, device=self.device)
            print(f'pos_encs.pe.shape: {self.pos_encs.pe.shape}')

        self._init_weights()


    def forward(self, z):
        seq_len = len(self.prefix_tokens) + z.shape[0]
        context_len = 2*self.n_tokens+len(self.prefix_tokens)+1

        if seq_len < context_len:
            if self.arch == 'attention':
                mask = torch.ones(1, context_len, device=self.device)
                mask[:, seq_len:] = 0
                mask = mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(-1, self.n_heads, context_len, -1)
                padding_tokens = torch.zeros(context_len - z.shape[0] - len(self.prefix_tokens), z.shape[1], device=self.device)
                z = torch.cat([self.prefix_tokens, z, padding_tokens], dim=0).unsqueeze(0)
            elif self.arch == 'transformer':
                mask = torch.zeros(1, context_len, device=self.device)
                mask[:, seq_len:] = 1
                mask = mask.bool()
                logging.info(f'mask.shape: {mask.shape}')
                
                padding_tokens = torch.zeros(context_len - z.shape[0] - len(self.prefix_tokens), z.shape[1], device=self.device)
                z = torch.cat([self.prefix_tokens, z, padding_tokens], dim=0).unsqueeze(0)  
        else:
            mask = None        
            # z: [bsz, z_dim]
            z = torch.cat([self.prefix_tokens, z], dim=0).unsqueeze(0)

        # z: [1, (scale * n_proj_layers) + bsz, z_dim]
        if self.use_pos_encs:
            z = self.pos_encs(z)
        # z: [1, (scale * n_proj_layers) + bsz, z_dim]
        if self.arch == 'transformer':
            encodings = self.hypnet(z, src_key_padding_mask=mask).squeeze(0)
        else:
            encodings = self.hypnet(z, mask).squeeze(0)

        # encodings: [(scale * n_proj_layers) + bsz, hypnet_dim]
        prefix_encodings = encodings[:len(self.prefix_tokens),:]
        # prefix_encodings: [n_proj_layers, hypnet_dim]

        biases = [] if self.predict_bias else None

        a_weights, b_weights = [], []
        for idx, generator in enumerate(self.generators):
            weight = (self.alpha / self.rank) * generator(prefix_encodings[idx])
            # weight: [(lm_emb_dim + mm_emb_dim) * rank] or [(lm_emb_dim + lm_emb_dim) * rank]
            a_weight = weight[:self.a_dims[idx]]
            b_weight = weight[self.a_dims[idx]:(self.a_dims[idx]+self.b_dims[idx])]

            if idx == 0 and self.hypnet_dim > self.mm_emb_dim:
                a_weight = a_weight[:self.mm_emb_dim * self.rank]
            
            a_weights.append(a_weight)
            b_weights.append(b_weight)
            if self.predict_bias:
                bias = weight[(self.a_dims[idx]+self.b_dims[idx]):]
                biases.append(bias)
        
        return a_weights, b_weights, biases
    

    def _init_weights(self):
        nn.init.xavier_uniform_(self.prefix_tokens)

        for generator in self.generators:
            nn.init.xavier_uniform_(generator.weight)
            nn.init.zeros_(generator.bias)


class HyperNetWrapper(nn.Module):
    def __init__(self, hn_args, proj_args, lm_emb_dim, mm_emb_dim, n_tokens, device):
        super(HyperNetWrapper, self).__init__()
        self.device = device
        self.hn_args = hn_args
        self.proj_args = proj_args
        self.hypernet = HyperNetwork(hn_args, lm_emb_dim, mm_emb_dim, n_tokens, device)
        self.projector = Projector(proj_args, lm_emb_dim, mm_emb_dim, device)
        self.projector.load_model()
        self.generated_projector = None

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.projector.eval()
        return self

    def generate_projector(self, z):
        with torch.no_grad():
            a_weights, b_weights, biases = self.hypernet(z)
            generated_projector = self.projector.combine_lora(a_weights, b_weights, biases)
        self.generated_projector = generated_projector

    def generate_projector_from_multiple_adapters(self, zs):
        with torch.no_grad():
            all_a_weights = []
            all_b_weights = []
            all_biases = []
            
            for z in zs:
                a_weights, b_weights, biases = self.hypernet(z)
                all_a_weights.append(a_weights)
                all_b_weights.append(b_weights)
                if biases is not None:
                    all_biases.append(biases)
            
            avg_a_weights = []
            avg_b_weights = []
            avg_biases = None
            
            for idx in range(len(all_a_weights[0])):
                layer_a_weights = torch.stack([weights[idx] for weights in all_a_weights])
                layer_b_weights = torch.stack([weights[idx] for weights in all_b_weights])
                
                avg_a_weights.append(torch.mean(layer_a_weights, dim=0))
                avg_b_weights.append(torch.mean(layer_b_weights, dim=0))
            
            if len(all_biases) > 0:
                avg_biases = []
                for idx in range(len(all_biases[0])):
                    layer_biases = torch.stack([bias[idx] for bias in all_biases])
                    avg_biases.append(torch.mean(layer_biases, dim=0))
            
            generated_projector = self.projector.combine_lora(avg_a_weights, avg_b_weights, avg_biases)
        
        self.generated_projector = generated_projector
        
    def forward(self, x, z):
        if self.generated_projector is not None:
            out = self.generated_projector(x)
        else:
            a_weights, b_weights, biases = self.hypernet(z)
            out = self.projector.lora_forward(x, a_weights, b_weights, biases)
        return out
    
    def trainable_parameters(self):
        if self.generated_projector is not None:
            return self.generated_projector.parameters()
        else:
            return self.hypernet.parameters()