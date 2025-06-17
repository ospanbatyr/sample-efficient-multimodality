import random
import pickle
import datasets
import os.path as osp
from dmi.data.base import BaseHypnetLoader

PATH = 'data/chebi20'

class CHEBI20Loader(BaseHypnetLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct):
        self.max_new_tokens = 401
        self.dataset_name = 'chebi20'
        self.PATH = 'data/chebi20' 
        self.modality = 'molecule'
        self.id_type = 'molid'
        self.emb_name = 'emb'
        self.CAPS_PER_IMAGE = 1
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self._init_datasets()
        self._init_prefix_emb_dict()

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
            item = {self.id_type:key, 'caption': value['caption'], 'smiles': value['smiles'], 'emb': value[self.emb_name]}
            self._update_means(item, split, n_items, text_emb_dict=text_emb_dict)
            split_set.append(item)          

        self._postprocess_means(split)
        split_set = datasets.Dataset.from_list(split_set)
        return split_set
        
    def instruct_tokenize(self, example):
        prefix = random.choice(self.prefixes)
        if self.is_instruct:
            texts = []
            for caption, smiles in zip(example["caption"], example["smiles"]):
                chat = [
                    {"role": "user", "content": f'{prefix}{smiles}'},
                    {"role": "assistant", "content": caption}
                ]
                texts.append(chat)
            text_input = self.tokenizer.apply_chat_template(texts, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False)                        
        else:
            raise NotImplementedError("Not implemented for non-instruct mode")

        return text_input, prefix



def max_token_length():
    PATH = 'chebi20'
    enc_name = 'MolCA'
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

                print(caption)
                print(f'Split: {split}, tokenizer name: {tokenizer_names[i]}, Max token length: {max_len}')

    print(max_len)

                
if __name__ == "__main__":
    max_token_length()