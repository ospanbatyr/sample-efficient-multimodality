import torch
import pickle
import datasets
import numpy as np
import os.path as osp
from dmi.data.base import BaseLoader

PATH = 'data/openvid'

class OpenvidLoader(BaseLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct, extract_features=False):
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self.PREFIX = 'Describe the video'
        self.extract_features = extract_features
        self.max_new_tokens = 77
        self._init_datasets()
        self.dataset_name = 'openvid'
    
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
            item = dict(videoid=key, caption=value['caption'], emb=value['emb'])

            if self.feed_txt_embs:
                text_emb = text_emb_dict[(int(key.split('_')[0]), value['caption'])]
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
        
        train_remove_columns = ['caption', 'videoid']
        eval_remove_columns = ['caption']
        self.train_set = train_set.map(self.tokenize_function, batched=True, num_proc=1, remove_columns=train_remove_columns)
        self.eval_set = validation_set.map(self.tokenize_function, batched=True, num_proc=1, remove_columns=eval_remove_columns)
    
    def train_collate(self, data):
        embs = [torch.FloatTensor(item['emb'][0]) for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        del new_data["emb"]
        collated_data = self.collator(new_data)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs

    def eval_collate(self, data):
        embs = [torch.FloatTensor(item['emb'][0]) for item in data]
        ids = [item['videoid'] for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        del new_data["emb"]
        collated_data = self.collator(new_data)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs, ids
    
    def subset_collate(self, data):
        embs = [torch.FloatTensor(item['emb'][0]) for item in data]
        embs = torch.stack(embs, dim=0)
        if self.subtract_mean:
            embs = embs - self.emb_mean
        
        if self.feed_txt_embs:
            text_embs = [torch.FloatTensor(item['text_emb']) for item in data]
            text_embs = torch.stack(text_embs, dim=0)
            if self.subtract_mean:
                text_embs = text_embs - self.text_emb_mean
                    
            new_embs = (embs, text_embs)
        else:
            new_embs = embs

        return new_embs
    

def postprocess_features():
    from nltk.tokenize import sent_tokenize
    PATH = 'openvid'
    enc_name = 'VideoCLIP-XL'
    splits = ['train', 'validation']

    for split in splits:
        with open(osp.join(PATH, f'{split}_embs_{enc_name}.pkl'), 'rb') as f:
            split_set_dict = pickle.load(f)

        for _, data in split_set_dict.items():
            data['caption'] = ' '.join(sent_tokenize(data['caption'])[:2])
            
        with open(osp.join(PATH, f'{split}_embs_{enc_name}_new.pkl'), 'wb') as f:
            pickle.dump(split_set_dict, f) 

def max_token_length():
    PATH = 'openvid'
    enc_name = 'VideoCLIP-XL'
    splits = ['validation']
    tokenizer_names = ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct']
    from transformers import AutoTokenizer
    tokenizers = [
        AutoTokenizer.from_pretrained(tokenizer_name) for tokenizer_name in tokenizer_names
    ]

    max_len = 0
    for split in splits:
        for i, tokenizer in enumerate(tokenizers):
                with open(osp.join(PATH, f'{split}_embs_{enc_name}_new.pkl'), 'rb') as f:
                    split_set_dict = pickle.load(f)
                
                for key, value in split_set_dict.items():
                    caption = value['caption']
                    tokens = tokenizer(caption, return_tensors='pt')['input_ids']
                    max_len = max(max_len, tokens.shape[1])

                print(caption)
                print(f'Split: {split}, tokenizer name: {tokenizer_names[i]}, Max token length: {max_len}')

    print(max_len)

                
if __name__ == "__main__":
    postprocess_features()
    max_token_length()