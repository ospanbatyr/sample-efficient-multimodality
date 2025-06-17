import os
import sys
import torch
import pickle
import datasets
import numpy as np
from tqdm import tqdm
import os.path as osp
from dmi.data.base import BaseLoader
from transformers import AutoTokenizer, AutoProcessor, ClapAudioModel
    
PATH = 'data/audiocaps'

class AudioCapsLoader(BaseLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct, extract_features=False):
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self.PREFIX = 'Caption the audio'
        self.extract_features = extract_features
        self.max_new_tokens = 42
        self._init_datasets()
    
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
            item = dict(audioid=key, caption=value['caption'], emb=value['embs'])

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
        
        train_remove_columns = ['caption', 'audioid']
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
        ids = [item['audioid'] for item in data]
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

def postprocess_features(model_name):
    splits = ["train", "val"]

    for split in splits:
        with open(f'audiocaps/{split}_embs_{model_name}.pkl', 'rb') as f:
            split_embs = pickle.load(f)

        for ids, data in split_embs.items():
            data['embs'] = data['embs'].cpu().numpy()
            data['caption'] = data['caption'][0]
            
        with open(f'audiocaps/{split}_embs_{model_name}_new.pkl', 'wb') as f:
            pickle.dump(split_embs, f) 
    

def extract_features_2(model_name):
    splits = ["train", "val"]
    if 'clap' == model_name:
        model = ClapAudioModel.from_pretrained("clap-htsat-fused").to('cuda')
        processor = AutoProcessor.from_pretrained("clap-htsat-fused")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model.eval()

    os.makedirs("audiocaps", exist_ok=True)

    loader = AudioCapsLoader(tokenizer, 1, 1, 1, 1, False, True)
    train_loader, eval_loader = loader.build_extract_loaders()

    loader_dict = dict(train=train_loader, val=eval_loader)
    for split in splits:
        loader = loader_dict[split]
        split_embs = dict()

        for i, batch in enumerate(tqdm(loader)):
            audios, captions, sampling_rate, audio_ids = batch["audios"], batch["captions"], batch["sampling_rate"], batch["ids"]
            inputs = processor(audios=audios, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)

            for j, audio_id in enumerate(audio_ids):
                if 'clap' == model_name:
                    split_embs[audio_id] = dict(caption=captions[j], embs=outputs.pooler_output[j])

            if i % 5000 == 4999:
                with open(f'audiocaps/{split}_embs_{model_name}.pkl', 'wb') as f:
                    pickle.dump(split_embs, f)

        with open(f'audiocaps/{split}_embs_{model_name}_new.pkl', 'wb') as f:
            pickle.dump(split_embs, f)



def check_audio():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    model = ClapAudioModel.from_pretrained("clap-htsat-fused").to('mps')
    processor = AutoProcessor.from_pretrained("clap-htsat-fused")

    loader_mgr = AudioCapsLoader(tokenizer, 2, 2, 2, 2, True)

    train_loader, eval_loader = loader_mgr.build_loaders()

    sampling_rates = []

    for batch in tqdm(train_loader):
        input_ids, attention_mask, labels, audios, sampling_rate = batch
        input_ids = input_ids.to('mps')
        attention_mask = attention_mask.to('mps')
        labels = labels.to('mps')

        sampling_rates.append(sampling_rate)

    import matplotlib.pyplot as plt
    plt.hist(sampling_rates, bins=100)
    plt.show()


def max_token_length():
    PATH = 'audiocaps'
    enc_name = 'clap-htsat-fused'
    splits = ['validation', 'test']
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

                print(f'Split: {split}, tokenizer name: {tokenizer_names[i]}, Max token length: {max_len}')

    print(max_len)

                
if __name__ == "__main__":
    max_token_length()