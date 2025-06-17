import torch
import torch.nn as nn

class HypernetMMModel(nn.Module):
    def __init__(self, llm, hypernet, device, mm_emb_dim, name, pad_token_id):
        super().__init__()
        self.llm = llm
        self.hypernet = hypernet
        self.device = device
        self.name = name
        self.pad_token_id = pad_token_id
        
        self.llm_dim = self.llm.config.hidden_size
        self.mm_emb_dim = mm_emb_dim

        self.embeddings = self.llm.get_input_embeddings()

        for param in self.llm.parameters():
            param.requires_grad = False

    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.llm.eval()
        return self
        
    def forward(self, mm_embeds, mm_stat_embeds, input_ids, attention_masks, labels):
        bsz = mm_embeds.shape[0]
        z = mm_stat_embeds
      
        out_embeds = self.hypernet(mm_embeds, z)
        
        text_embeds = self.embeddings(input_ids) # [bsz, text_len, lm_dim]

        projected_embeds = out_embeds.unsqueeze(1)

        input_embeds = torch.cat((projected_embeds, text_embeds), dim=1) # [bsz, mm_len + text_len, lm_dim]
        
        new_attention_part = torch.ones(bsz, 1).to(self.device) # [bsz, 1]
        attention_masks = torch.cat((new_attention_part, attention_masks), dim=-1)

        new_label_part = torch.full((bsz, 1), -100).to(self.device)
        labels = torch.cat((new_label_part, labels), dim=-1)

        def f_llm():
            return self.llm(inputs_embeds=input_embeds, labels=labels)

        if self.device == 'cuda':
            with torch.amp.autocast(self.device):
                outputs = f_llm()
        else:
            outputs = f_llm()

        return outputs.loss, out_embeds

    def generate(self, mm_embeds, mm_subset_embeds, max_new_tokens, prefix=None):
        out_embeds = self.hypernet(mm_embeds, mm_subset_embeds)
        projected_embeds = out_embeds.unsqueeze(1)

        if prefix is not None:
            prefix_embeds = self.embeddings(prefix)
            input_embeds = torch.cat((projected_embeds, prefix_embeds), dim=1)
        else:
            input_embeds = projected_embeds

        def f_llm():
            return self.llm.generate(inputs_embeds=input_embeds, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id)

        with torch.no_grad():
            if self.device == 'cuda':
                with torch.amp.autocast(self.device):
                    outputs = f_llm()
            else:
                outputs = f_llm()

        return outputs


class ProjectorMMModel(nn.Module):
    def __init__(self, llm, projector, device, mm_emb_dim, name, pad_token_id):
        super().__init__()
        self.llm = llm
        self.projector = projector
        self.device = device
        self.name = name
        
        self.llm_dim = self.llm.config.hidden_size
        self.mm_emb_dim = mm_emb_dim
        self.pad_token_id = pad_token_id

        self.embeddings = self.llm.get_input_embeddings()

        for param in self.llm.parameters():
            param.requires_grad = False

    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.llm.eval()
        return self
        
    def forward(self, mm_embeds, input_ids, attention_masks, labels):
        bsz = mm_embeds.shape[0]
        # mm_embeds: [bsz, mm_emb_dim]
        # input_ids: [bsz, text_len]
        # attention_masks: [bsz, text_len]
 
        out_embeds = self.projector(mm_embeds)
        # out_embeds: [bsz, lm_emb_dim]
        
        text_embeds = self.embeddings(input_ids) # [bsz, text_len, lm_dim]
        # text_embeds: [bsz, text_len, lm_dim]

        projected_embeds = out_embeds.unsqueeze(1)
        # projected_embeds: [bsz, 1, lm_emb_dim]

        input_embeds = torch.cat((projected_embeds, text_embeds), dim=1) # [bsz, mm_len + text_len, lm_dim]
        # input_embeds: [bsz, (text_len+1), lm_emb_dim]

        new_attention_part = torch.ones(bsz, 1).to(self.device) # [bsz, 1]
        attention_masks = torch.cat((new_attention_part, attention_masks), dim=-1)
        # attention_masks: [bsz, (text_len+1)]
        
        new_label_part = torch.full((bsz, 1), -100).to(self.device)
        labels = torch.cat((new_label_part, labels), dim=-1)
        # labels: [bsz, (text_len+1)]

        def f_llm():
            return self.llm(inputs_embeds=input_embeds, labels=labels)

        if self.device == 'cuda':
            with torch.amp.autocast(self.device):
                outputs = f_llm()
        else:
            outputs = f_llm()

        return outputs.loss
    
    def generate(self, mm_embeds, max_new_tokens, prefix=None):      
        out_embeds = self.projector(mm_embeds) # [bsz, lm_emb_dim]
        projected_embeds = out_embeds.unsqueeze(1) # [bsz, 1, lm_emb_dim]

        if prefix is not None:
            prefix_embeds = self.embeddings(prefix)
            input_embeds = torch.cat((projected_embeds, prefix_embeds), dim=1)
        else:
            input_embeds = projected_embeds
        
        def f_llm():
            return self.llm.generate(inputs_embeds=input_embeds, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id)

        with torch.no_grad():
            if self.device == 'cuda':
                with torch.amp.autocast(self.device):
                    outputs = f_llm()
            else:
                outputs = f_llm()

        return outputs
    

class LoraMMModel(nn.Module):
    def __init__(self, llm, lora_model, device, mm_emb_dim, name, pad_token_id):
        super().__init__()
        self.llm = llm
        self.lora_model = lora_model
        self.device = device
        self.name = name
        self.pad_token_id = pad_token_id
        
        self.llm_dim = self.llm.config.hidden_size
        self.mm_emb_dim = mm_emb_dim

        self.embeddings = self.llm.get_input_embeddings()

        for param in self.llm.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.llm.eval()
        return self
    
    def forward(self, mm_embeds, input_ids, attention_masks, labels):
        bsz = mm_embeds.shape[0]
        # mm_embeds: [bsz, mm_emb_dim]
        # input_ids: [bsz, text_len]
        # attention_masks: [bsz, text_len]

        out_embeds = self.lora_model(mm_embeds)
        # out_embeds: [bsz, lm_emb_dim]

        text_embeds = self.embeddings(input_ids) 
        # text_embeds: [bsz, text_len, lm_dim]

        projected_embeds = out_embeds.unsqueeze(1)
        # projected_embeds: [bsz, 1, lm_emb_dim]

        input_embeds = torch.cat((projected_embeds, text_embeds), dim=1)
        # input_embeds: [bsz, (text_len+1), lm_dim]

        new_attention_part = torch.ones(bsz, 1).to(self.device)
        attention_masks = torch.cat((new_attention_part, attention_masks), dim=-1)

        new_label_part = torch.full((bsz, 1), -100).to(self.device)
        labels = torch.cat((new_label_part, labels), dim=-1)
        # labels: [bsz, (text_len+1)]

        def f_llm():
            return self.llm(inputs_embeds=input_embeds, labels=labels)

        if self.device == 'cuda':
            with torch.amp.autocast(self.device):
                outputs = f_llm()
        else:
            outputs = f_llm()

        return outputs.loss

    def generate(self, mm_embeds, max_new_tokens, prefix=None):      
        out_embeds = self.lora_model(mm_embeds) # [bsz, lm_emb_dim]
        projected_embeds = out_embeds.unsqueeze(1) # [bsz, 1, lm_emb_dim]

        if prefix is not None:
            prefix_embeds = self.embeddings(prefix)
            input_embeds = torch.cat((projected_embeds, prefix_embeds), dim=1)
        else:
            input_embeds = projected_embeds
        
        def f_llm():
            return self.llm.generate(inputs_embeds=input_embeds, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id)

        with torch.no_grad():
            if self.device == 'cuda':
                with torch.amp.autocast(self.device):
                    outputs = f_llm()
            else:
                outputs = f_llm()

        return outputs

