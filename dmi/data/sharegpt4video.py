import torch
import random
import pickle
import datasets
import numpy as np
import os.path as osp
from dmi.data.base import BaseHypnetLoader

PATH = 'data/sharegpt4video'

class ShareGPT4VideoLoader(BaseHypnetLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct, extract_features=False):
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self.extract_features = extract_features
        self.max_new_tokens = 605
        self._init_datasets()
        self._init_prefix_emb_dict()
        self.dataset_name = 'sharegpt4video'

    def _init_prefix_emb_dict(self):
        with open(osp.join('data/prefixes/video_inst.pkl'), 'rb') as f:
            self.prefix_emb_dict = pickle.load(f)
            self.prefixes = list(self.prefix_emb_dict.keys())
    
    def _init_split(self, split):
        with open(osp.join(PATH, f'{split}_embs_{self.model_name}.pkl'), 'rb') as f:
            split_set_dict = pickle.load(f)

        split_set = []

        if self.feed_txt_embs:
            with open(osp.join(PATH, f'{split}_embs_gte-modernbert-base.pkl'), 'rb') as f:
                text_emb_dict = pickle.load(f)
        
        if self.subtract_mean and split == 'train':
            n = 0
            self.emb_mean = None
            if self.feed_txt_embs:
                self.text_emb_mean = None

        for idx, (key, value) in enumerate(split_set_dict.items()):
            item = dict(videoid=key, caption=value['caption'], emb=value['embs'])

            if self.feed_txt_embs:
                text_emb = text_emb_dict[(key, value['caption'])]
                item['text_emb'] = text_emb
            
            if self.subtract_mean and split == 'train':
                if self.emb_mean is None:
                    self.emb_mean = item['emb']
                    if self.feed_txt_embs:
                        self.text_emb_mean = item['text_emb']
                else:
                    n += 1
                    self.emb_mean = self.emb_mean + (item['emb'] - self.emb_mean) / (n + 1)
                    if self.feed_txt_embs:
                        self.text_emb_mean = self.text_emb_mean + (item['text_emb'] - self.text_emb_mean) / (n + 1)
                    
            split_set.append(item)          

        if self.subtract_mean and split == 'train':
            self.emb_mean = self.emb_mean[np.newaxis, :]
            if self.feed_txt_embs:
                self.text_emb_mean = self.text_emb_mean[np.newaxis, :]
        
        split_set = datasets.Dataset.from_list(split_set)
        return split_set
    
    def _init_datasets(self):
        train_set = self._init_split('train')
        validation_set = self._init_split('validation')

        if self.debug:
            train_set = train_set.select(range(4*self.train_batch_size))
            validation_set = validation_set.select(range(4*self.eval_batch_size))
        elif self.dataset_size != "full":
            dataset_size = int(self.dataset_size)
            train_set = train_set.shuffle(seed=self.seed).select(range(dataset_size))
        
        print(f"Using {self.dataset_size} samples, {len(train_set)} samples in the subset")
        
        self.train_set = train_set
        self.eval_set = validation_set
    
    def instruct_tokenize(self, example):
        prefix = random.choice(self.prefixes)
        if self.is_instruct:
            captions = []
            for caption in example["caption"]:
                chat = [
                    {"role": "user", "content": prefix},
                    {"role": "assistant", "content": caption}
                ]
                captions.append(chat)
            text_input = self.tokenizer.apply_chat_template(captions, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False)                        
        else:
            captions = [name for name in example["caption"]]
            text_input = self.tokenizer(captions)
        return text_input, prefix

    def train_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
        embs = torch.stack(embs, dim=0)

        new_data = {key: [d[key] for d in data] for key in data[0]}
        text_input, _ = self.instruct_tokenize(new_data)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        collated_data = self.collator(text_input)
        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs

    def eval_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
        ids = [item['videoid'] for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        text_input, _ = self.instruct_tokenize(new_data)
        collated_data = self.collator(text_input)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs, ids
    
    def subset_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        prefix = random.choice(self.prefixes)
        if self.feed_txt_embs:
            prefix_emb = torch.FloatTensor(self.prefix_emb_dict[prefix]).unsqueeze(0)
            text_embs = [torch.FloatTensor(item['text_emb']) for item in data]
            text_embs = torch.stack(text_embs, dim=0)
            if self.subtract_mean:
                text_embs = text_embs - self.text_emb_mean
                    
            new_embs = (embs, text_embs, prefix_emb)
        else:
            new_embs = embs

        return new_embs

def max_token_length():
    PATH = 'sharegpt4video'
    enc_name = 'ViCLIP-B-16'
    splits = ['validation']
    tokenizer_names = ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct']
    from transformers import AutoTokenizer
    tokenizers = [
        AutoTokenizer.from_pretrained(tokenizer_name) for tokenizer_name in tokenizer_names
    ]

    max_len = 0
    for split in splits:
        for i, tokenizer in enumerate(tokenizers):
                with open(osp.join(PATH, f'{split}_embs_{enc_name}.pkl'), 'rb') as f:
                    split_set_dict = pickle.load(f)
                
                for key, value in split_set_dict.items():
                    caption = value['caption']
                    tokens = tokenizer(caption, return_tensors='pt')['input_ids']
                    max_len = max(max_len, tokens.shape[1])

                print(caption)
                print(f'Split: {split}, tokenizer name: {tokenizer_names[i]}, Max token length: {max_len}')

    print(max_len)

                
if __name__ == "__main__":
    max_token_length()