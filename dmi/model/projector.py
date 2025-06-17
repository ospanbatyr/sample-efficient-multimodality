import torch
import torch.nn as nn
from dmi.utils.args import ProjectorArgs, setup_args


class Projector(nn.Module):
    def __init__(
            self, projector_args: ProjectorArgs, lm_emb_dim, mm_emb_dim, device
        ):
        super().__init__()
        self.lm_emb_dim = lm_emb_dim
        self.mm_emb_dim = mm_emb_dim
        self.device = device
        setup_args(self, prefix='proj_', args=projector_args)
        self.build_model()

    def act_function(self):
        if self.act == 'quick_gelu':
            act = nn.GELU
        else:
            raise NotImplementedError
        return act

    def build_model(self):
        modules = []
        if self.arch == 'linear':
            modules.append(nn.Linear(self.mm_emb_dim, self.lm_emb_dim))
            modules.append(nn.Dropout(self.dropout))
        elif self.arch == 'mlp':
            assert self.n_layers >= 2, f'MLP should at least have depth of two, cur depth = {self.n_layers}'
            modules.append(nn.Linear(self.mm_emb_dim, self.lm_emb_dim))
            modules.append(self.act_function()(approximate='tanh'))
            modules.append(nn.Dropout(self.dropout))
                
            for _ in range(self.n_layers - 2):
                modules.append(nn.Linear(self.lm_emb_dim, self.lm_emb_dim))
                modules.append(self.act_function()(approximate='tanh'))
                modules.append(nn.Dropout(self.dropout))
            
            modules.append(nn.Linear(self.lm_emb_dim, self.lm_emb_dim))
        else:
            raise NotImplementedError
        
        self.net = nn.ModuleList(modules)
    
    def load_model(self):
        assert self.name_or_path is not None
        checkpoint = torch.load(self.name_or_path, map_location=self.device)
        if self.prune is not None:
            for layer in checkpoint['projector_state_dict']:
                if 'net.0.weight' in layer:
                    checkpoint['projector_state_dict'][layer] = checkpoint['projector_state_dict'][layer][:, :self.prune]
            
        self.load_state_dict(checkpoint['projector_state_dict'])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    
    def only_lora_forward(self, x, loras):
        linear_idx = 0  # Track position in weight arrays
        for layer in self.net:
            layer_out = layer(x)
            if layer.__class__.__name__ != 'Linear':
                x = layer_out
                continue
            
            lora = loras[linear_idx]
            linear_idx += 1
            # x: [bsz, mm_emb_dim] or [bsz, lm_emb_dim]
            x = layer_out + lora(x)

        return x

    def combine_lora(self, a_weights, b_weights, biases):
        if biases is None:
            biases = [torch.zeros(self.lm_emb_dim).to(self.device) for _ in range(len(a_weights))]

        modules = []
        linear_idx = 0  # Track position in weight arrays

        for layer in self.net:
            if layer.__class__.__name__ != 'Linear':
                modules.append(layer)
                continue
            
            if linear_idx >= len(a_weights):
                raise ValueError("Not enough weights provided for all linear layers")
                
            w_a = a_weights[linear_idx]
            w_b = b_weights[linear_idx]
            b = biases[linear_idx]

            w_a = w_a.reshape(layer.in_features, -1)
            w_b = w_b.reshape(-1, layer.out_features)

            # w_a: [mm_emb_dim, rank] or [lm_emb_dim, rank]
            # w_b: [rank, lm_emb_dim]
            # b: [lm_emb_dim]

            combined_weights = (w_a @ w_b).T + torch.clone(layer.weight)
            combined_biases = b + torch.clone(layer.bias)
            # combined_weights: [mm_emb_dim, lm_emb_dim] or [lm_emb_dim, lm_emb_dim]
            # combined_biases: [lm_emb_dim]

            linear = nn.Linear(combined_weights.shape[0], combined_weights.shape[1])
            linear.weight = nn.Parameter(combined_weights)
            linear.bias = nn.Parameter(combined_biases)
            modules.append(linear)
            linear_idx += 1

        if linear_idx < len(a_weights):
            raise ValueError("Too many weights provided")

        return nn.Sequential(*modules).to(self.device)
    
    def lora_forward(self, x, a_weights, b_weights, biases):
        bsz = x.shape[0]

        if biases is None:
            biases = [torch.zeros(self.lm_emb_dim).to(self.device) for _ in range(len(a_weights))]

        for layer, w_a, w_b, b in zip(self.net, a_weights, b_weights, biases):
            layer_out = layer(x)
            if layer.__class__.__name__ != 'Linear':
                x = layer_out
                continue
    
            x_expanded = x.unsqueeze(1)
            # b: [lm_emb_dim]
            # x: [bsz, mm_emb_dim] or [bsz, lm_emb_dim]
            # x_expanded: [bsz, 1, mm_emb_dim] or [bsz, 1, lm_emb_dim]

            # w_a: [mm_emb_dim * rank] or [lm_emb_dim * rank]
            # w_b: [rank * lm_emb_dim]
            w_a = w_a.reshape(layer.in_features, -1)
            w_b = w_b.reshape(-1, layer.out_features)

            # w_a: [mm_emb_dim, rank] or [lm_emb_dim, rank]
            # w_b: [rank, lm_emb_dim]

            w_a = w_a.unsqueeze(0).expand(bsz, -1, -1)
            w_b = w_b.unsqueeze(0).expand(bsz, -1, -1)

            # w_a: [bsz, mm_emb_dim, rank] or [bsz, lm_emb_dim, rank]
            # w_b: [bsz, rank, lm_emb_dim]

            inter = torch.bmm(x_expanded, w_a)
            # inter: [bsz, 1, rank]

            out = torch.bmm(inter, w_b).squeeze(1)
            # out: [bsz, lm_emb_dim]
            
            out = out + b

            x = layer_out + out

        return x
