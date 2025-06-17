import torch
import random
import pickle
import logging
import datasets
import numpy as np
import os.path as osp
from .inffs import InfFS
from copy import deepcopy
from functools import partial
from torch.utils.data import DataLoader
from dmi.utils.sampler import InfiniteSampler

def datacollator(tokenizer, is_instruct, model_inputs):
    batch_size = len(model_inputs['input_ids'])
    model_inputs["labels"] = deepcopy(model_inputs["input_ids"])

    for i in range(batch_size):
        model_inputs['input_ids'][i] = model_inputs['input_ids'][i] + [tokenizer.eos_token_id]
        model_inputs["labels"][i] = model_inputs["labels"][i] + [tokenizer.eos_token_id]
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i]) 

        if is_instruct:
            model_inputs["assistant_masks"][i] = model_inputs["assistant_masks"][i] + [1]

            for j in range(len(model_inputs["assistant_masks"][i])):
                if model_inputs["assistant_masks"][i][j] == 0:      # if the token is not an assistant token
                    model_inputs["labels"][i][j] = -100             # don't optimize on it

    if is_instruct:
        del model_inputs["assistant_masks"]

    max_length = max(len(input_ids) for input_ids in model_inputs['input_ids'])

    for i in range(batch_size):
        sample_input_ids = model_inputs['input_ids'][i]
        sample_labels = model_inputs['labels'][i]

        if tokenizer.padding_side == "right":
            model_inputs['input_ids'][i] = sample_input_ids + \
                [tokenizer.pad_token_id] * (max_length - len(sample_input_ids))
            model_inputs['labels'][i] = sample_labels + \
                [tokenizer.pad_token_id] * (max_length - len(sample_labels))
            model_inputs['attention_mask'][i] = model_inputs['attention_mask'][i] + \
                [0] * (max_length - len(model_inputs['attention_mask'][i]))
        elif tokenizer.padding_side ==  "left":
            model_inputs['input_ids'][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) \
                + sample_input_ids
            model_inputs['labels'][i] = [tokenizer.pad_token_id] * (max_length - len(sample_labels)) \
                + sample_labels
            model_inputs['attention_mask'][i] = [0] * (max_length - len(model_inputs['attention_mask'][i])) \
                + model_inputs['attention_mask'][i]

        model_inputs['input_ids'][i] = torch.tensor(model_inputs['input_ids'][i][:max_length])
        model_inputs['attention_mask'][i] = torch.tensor(model_inputs['attention_mask'][i][:max_length])
        model_inputs['labels'][i] = torch.tensor(model_inputs['labels'][i][:max_length])

    model_inputs['input_ids'] = torch.stack(model_inputs['input_ids'], dim=0)
    model_inputs['attention_mask'] = torch.stack(model_inputs['attention_mask'], dim=0)
    model_inputs['labels'] = torch.stack(model_inputs['labels'], dim=0)
 
    return model_inputs


class BaseLoader:
    def __init__(self, tokenizer, train_args, model_name, is_instruct, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.train_args = train_args
        self.pad_to_multiple_of = self.train_args.pad_to_multiple_of
        self.train_batch_size = self.train_args.train_batch_size
        self.eval_batch_size = self.train_args.eval_batch_size
        self.subset_batch_size = self.train_args.subset_batch_size
        self.n_components = self.train_args.n_components
        self.is_instruct = is_instruct
        self.debug = self.train_args.debug
        self.model_name = model_name
        self.feed_txt_embs = self.train_args.feed_txt_embs
        self.dataset_size = self.train_args.dataset_size
        self.subtract_mean = self.train_args.subtract_mean
        self.seed = self.train_args.seed
        self.args = args
        self.kwargs = kwargs
        self.collator = partial(datacollator, tokenizer, is_instruct)

    def tokenize_function(self, example):
        if self.is_instruct:
            captions = []
            for caption in example["caption"]:
                chat = [
                    {"role": "user", "content": self.PREFIX},
                    {"role": "assistant", "content": caption}
                ]
                captions.append(chat)
            text_input = self.tokenizer.apply_chat_template(captions, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False)                        
        else:
            captions = [name for name in example["caption"]]
            text_input = self.tokenizer(captions)
        return text_input

    def _select_features(self, split_set_dict):
        inf = InfFS()
        all_embs = np.array([item[self.emb_name] for item in split_set_dict.values()])
        RANKED, _ = inf.infFS(all_embs, y_train=None, alpha=0.2, supervision=False, verbose=True) # alpha taken from 2015 paper
        self.selected_features = list(RANKED[:self.n_components]) # RANKED is flipped argsort, so we take the first n_comps
    
    def _init_means(self, split):
        if split == 'train' and self.subtract_mean:
            self.emb_mean = None
            if self.feed_txt_embs:
                self.text_emb_mean = None

    def _update_means(self, item, split, n, text_emb_dict=None):
        if self.feed_txt_embs:
            text_emb = text_emb_dict[(item[self.id_type], item['caption'])]
            item['text_emb'] = text_emb
        
        if split == 'train' and self.subtract_mean:
            if self.emb_mean is None:
                self.emb_mean = item['emb']
                if self.feed_txt_embs:
                    self.text_emb_mean = item['text_emb']
            else:
                n += 1
                self.emb_mean = self.emb_mean + (item['emb'] - self.emb_mean) / (n + 1)
                if self.feed_txt_embs:
                    self.text_emb_mean = self.text_emb_mean + (item['text_emb'] - self.text_emb_mean) / (n + 1)

    def _init_prefix_emb_dict(self):
        with open(osp.join(f'data/prefixes/{self.modality}_inst.pkl'), 'rb') as f:
            self.prefix_emb_dict = pickle.load(f)
            self.prefixes = list(self.prefix_emb_dict.keys())

    def _postprocess_means(self, split):
        if split == 'train' and self.subtract_mean:
            self.emb_mean = self.emb_mean[np.newaxis, :]
            if self.feed_txt_embs:
                self.text_emb_mean = self.text_emb_mean[np.newaxis, :]

    def _subsample_dataset(self, split_set_dict, text_emb_dict=None):
        dataset_size = int(self.dataset_size)
        baseid_set = set()

        for _, (cur_id, _) in enumerate(split_set_dict.items()):
            baseid = cur_id.split('_')[0]
            if len(baseid_set) <= (dataset_size // self.CAPS_PER_IMAGE):
                baseid_set.add(baseid)

        split_set_dict = {k: v for k, v in split_set_dict.items() if k.split('_')[0] in baseid_set}
        split_set_dict = {k: v for i, (k, v) in enumerate(split_set_dict.items()) if i < dataset_size}

        # print ids in the split_set_dict
        logging.info(split_set_dict.keys())

        if self.feed_txt_embs:
            text_emb_dict = {k: v for k, v in text_emb_dict.items() if k[0] in split_set_dict}

        return split_set_dict, text_emb_dict

    def _init_split(self, split):
        with open(osp.join(self.PATH, f'{split}_embs_{self.model_name}.pkl'), 'rb') as f:
            split_set_dict = pickle.load(f)

        if self.feed_txt_embs:
            with open(osp.join(self.PATH, f'{split}_embs_gte-modernbert-base.pkl'), 'rb') as f:
                text_emb_dict = pickle.load(f)
        else:
            text_emb_dict = None

        if self.dataset_size != "full" and split == 'train':
            split_set_dict, text_emb_dict = self._subsample_dataset(split_set_dict, text_emb_dict=text_emb_dict)

        if split == 'train' and self.n_components is not None:
            self._select_features(split_set_dict)

        self._init_means(split)
        split_set = []
        n_items = 0
        for key, value in split_set_dict.items():
            item = {self.id_type:key, 'caption': value['caption'], 'emb': value[self.emb_name]}
            self._update_means(item, split, n_items, text_emb_dict=text_emb_dict)
            split_set.append(item)          
        self._postprocess_means(split)

        split_set = datasets.Dataset.from_list(split_set)
        return split_set

    def _init_datasets(self):
        train_set = self._init_split('train')
        validation_set = self._init_split('validation')
        test_set = self._init_split('test')

        if self.debug:
            train_set = train_set.select(range(4*self.train_batch_size))
            validation_set = validation_set.select(range(4*self.eval_batch_size))
            test_set = test_set.select(range(4*self.eval_batch_size))

        print(f"Using {self.dataset_size} samples, {len(train_set)} samples in the subset")
        
        #validation_set = validation_set.select(range(16*self.eval_batch_size))
        #test_set = test_set.select(range(16*self.eval_batch_size))
        self.train_set = train_set
        self.eval_set = validation_set
        self.test_set = test_set

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
        if self.n_components is not None:
            embs = []
            for item in data:
                embs.append(torch.FloatTensor(item['emb'])[self.selected_features])
        else:
            embs = [torch.FloatTensor(item['emb']) for item in data]

        embs = torch.stack(embs, dim=0)

        new_data = {key: [d[key] for d in data] for key in data[0]}
        text_input, _ = self.instruct_tokenize(new_data)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        collated_data = self.collator(text_input)
        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs

    def eval_collate(self, data):
        if self.n_components is not None:
            embs = []
            for item in data:
                embs.append(torch.FloatTensor(item['emb'])[self.selected_features])
        else:
            embs = [torch.FloatTensor(item['emb']) for item in data]

        ids = [item[self.id_type] for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        text_input, _ = self.instruct_tokenize(new_data)
        collated_data = self.collator(text_input)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs, ids
    
    def subset_collate(self, data):
        if self.n_components is not None:
            embs = []
            for item in data:
                embs.append(torch.FloatTensor(item['emb'])[self.selected_features])
        else:
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

    def build_hypnet_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, collate_fn=self.train_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed))
        train_subset_loader = DataLoader(self.train_set, batch_size=self.subset_batch_size, num_workers=0, collate_fn=self.subset_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed))

        eval_loader = DataLoader(self.eval_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        eval_subset_loader = DataLoader(self.eval_set, batch_size=self.subset_batch_size, num_workers=0, collate_fn=self.subset_collate, sampler=InfiniteSampler(length=len(self.eval_set), train_args=self.train_args, seed=self.seed))

        return train_loader, train_subset_loader, eval_loader, eval_subset_loader

    def build_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, collate_fn=self.train_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed))
        eval_loader = DataLoader(self.eval_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        return train_loader, eval_loader

    def build_test_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, collate_fn=self.train_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed))
        eval_loader = DataLoader(self.test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        return train_loader, eval_loader

    def build_eval_and_test_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, collate_fn=self.train_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed))
        eval_loader = DataLoader(self.eval_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        test_loader = DataLoader(self.test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        return train_loader, eval_loader, test_loader
    
    def build_fewshot_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, collate_fn=self.train_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed, bsz=self.train_batch_size))
        train_subset_loader = DataLoader(self.train_set, batch_size=self.subset_batch_size, num_workers=0, collate_fn=self.subset_collate, sampler=InfiniteSampler(length=len(self.train_set), train_args=self.train_args, seed=self.seed, bsz=self.subset_batch_size))

        eval_loader = DataLoader(self.eval_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        eval_subset_loader = DataLoader(self.eval_set, batch_size=self.subset_batch_size, num_workers=0, collate_fn=self.subset_collate, sampler=InfiniteSampler(length=len(self.eval_set), train_args=self.train_args, seed=self.seed, bsz=self.subset_batch_size))

        test_loader = DataLoader(self.test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0, collate_fn=self.eval_collate)
        test_subset_loader = DataLoader(self.test_set, batch_size=self.subset_batch_size, num_workers=0, collate_fn=self.subset_collate, sampler=InfiniteSampler(length=len(self.test_set), train_args=self.train_args, seed=self.seed, bsz=self.subset_batch_size))

        return train_loader, train_subset_loader, eval_loader, eval_subset_loader, test_loader, test_subset_loader


class BaseHypnetLoader(BaseLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct, *args, **kwargs) -> None:
        super().__init__(tokenizer, train_args, model_name, is_instruct, *args, **kwargs)

    def tokenize_function(self, *args):
        raise NotImplementedError("Subclasses must implement this method")
    
