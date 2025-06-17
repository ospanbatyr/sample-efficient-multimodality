import os
import json
import torch
import pickle
import string
import datasets
import numpy as np
from PIL import Image
import os.path as osp
from tqdm import tqdm
from dmi.data.base import BaseLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

PATH = "data/coco"

class COCODataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.path = '/datasets/COCO'
        self.split = split
        self.split_name = f'{self.split}2017'

        self.setup_dataset()

    def setup_dataset(self):
        self.image_dir = osp.join(self.path, self.split_name)
        self.annotation_file = osp.join(self.path, 'annotations', f'captions_{self.split_name}.json')

        with open(self.annotation_file, "r") as f:
            json_data = json.load(f)
            annotations = json_data['annotations']

        image_dict = dict()
        for item in json_data['images']:
            image_dict[item['id']] = item

        self.annotations = annotations
        self.image_dict = image_dict

    def __len__(self):
        return len(self.annotations)

    def _read_image_info(self, idx):
        image_id = self.annotations[idx]['image_id']
        file_name = self.image_dict[image_id]['file_name']
        file_path = osp.join(self.image_dir, file_name)
        return image_id, file_path

    def __getitem__(self, idx):
        image_id, file_path = self._read_image_info(idx)
        caption = self.annotations[idx]['caption'].strip(string.punctuation)
        return dict(img_id=image_id, file_path=file_path, caption=caption)


class COCOLoader(BaseLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct, extract_features=False):
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self.PREFIX = 'Caption the image'
        self.extract_features = extract_features
        self.max_new_tokens = 56
        self._init_datasets()
        self.dataset_name = 'coco'
    
    def _init_split(self, split):
        if self.extract_features:
            split_set = datasets.Dataset.from_list(COCODataset(split))
            return split_set
        else:
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
                item = dict(imageid=key, caption=value['caption'], emb=value['embs'])

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
        if self.extract_features:
            validation_set = self._init_split('val')
        else:
            validation_set = self._init_split('validation')

        if self.debug:
            train_set = train_set.select(range(4*self.train_batch_size))
            validation_set = validation_set.select(range(4*self.eval_batch_size))
        elif self.dataset_size != "full":
            dataset_size = int(self.dataset_size)
            train_set = train_set.shuffle(seed=self.seed).select(range(dataset_size))
        
        print(f"Using {self.dataset_size} samples, {len(train_set)} samples in the subset")
        
        if self.extract_features:
            self.train_set = train_set
            self.eval_set = validation_set
        else:
            train_remove_columns = ['caption', 'imageid']
            eval_remove_columns = ['caption']
            self.train_set = train_set.map(self.tokenize_function, batched=True, num_proc=1, remove_columns=train_remove_columns)
            self.eval_set = validation_set.map(self.tokenize_function, batched=True, num_proc=1, remove_columns=eval_remove_columns)
    
    def train_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        del new_data["emb"]
        collated_data = self.collator(new_data)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs

    def eval_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
        ids = [item['imageid'] for item in data]
        embs = torch.stack(embs, dim=0)

        if self.subtract_mean:
            embs = embs - self.emb_mean

        new_data = {key: [d[key] for d in data] for key in data[0]}
        del new_data["emb"]
        collated_data = self.collator(new_data)

        return collated_data['input_ids'], collated_data['attention_mask'], collated_data['labels'], embs, ids
    
    def subset_collate(self, data):
        embs = [torch.FloatTensor(item['emb']) for item in data]
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
    
    def extract_collate(self, data):
        images = [Image.open(e["file_path"]).convert('RGB') for e in data]
        ids = [item['img_id'] for item in data]
        captions = [item['caption'] for item in data]
        return images, captions, ids
    
    def build_extract_loaders(self):
        train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=0, shuffle=False, drop_last=False, pin_memory=True, collate_fn=self.extract_collate)
        eval_loader = DataLoader(self.eval_set, batch_size=self.eval_batch_size, num_workers=0, shuffle=False, drop_last=False, pin_memory=True, collate_fn=self.extract_collate)
        return train_loader, eval_loader


import timm
from transformers import CLIPVisionModelWithProjection, AutoProcessor, SiglipVisionModel

def extract_features(model_name):
    splits = ["train", "val"]
    if 'timm' in model_name:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to('cuda')
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        emb_name = model_name.split('/')[-1]
    elif 'clip' == model_name:
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        emb_name = "openai/clip-vit-large-patch14".split('/')[-1]
    elif 'siglip' == model_name:
        model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to('cuda')
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        emb_name = "google/siglip-base-patch16-224".split('/')[-1]


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model.eval()

    os.makedirs("coco", exist_ok=True)

    from dmi.utils.args import TrainArgs

    train_args = TrainArgs(output_dir=None, train_batch_size=17, eval_batch_size=17, subset_batch_size=17)
    loader = COCOLoader(tokenizer, train_args, model_name, is_instruct=False, extract_features=True)

    train_loader, eval_loader = loader.build_extract_loaders()

    loader_dict = dict(train=train_loader, val=eval_loader)
    for split in splits:
        loader = loader_dict[split]
        split_embs = dict()

        for i, batch in enumerate(tqdm(loader)):
            images, captions, img_ids = batch
            if 'timm' in model_name:
                images = [transforms(image) for image in images]
                images = torch.stack(images, dim=0).to('cuda')
                outputs = model(images)
            else:
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)

            for j, img_id in enumerate(img_ids):
                if 'timm' in model_name:
                    split_embs[f'{img_id}_{i}_{j}'] = dict(caption=captions[j], embs=outputs[j].cpu().detach())
                elif 'clip' == model_name:
                    split_embs[f'{img_id}_{i}_{j}'] = dict(caption=captions[j], embs=outputs.image_embeds[j].cpu().detach())
                elif 'siglip' == model_name:
                    split_embs[f'{img_id}_{i}_{j}'] = dict(caption=captions[j], embs=outputs.pooler_output[j].cpu().detach())

            if i % 5000 == 4999:
                print(f'coco/{split}_embs_{emb_name}.pkl')
                with open(f'coco/{split}_embs_{emb_name}.pkl', 'wb') as f:
                    pickle.dump(split_embs, f)

        with open(f'coco/{split}_embs_{emb_name}.pkl', 'wb') as f:
            pickle.dump(split_embs, f)


def max_token_length():
    PATH = 'coco'
    enc_name = 'dinov2-base'
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