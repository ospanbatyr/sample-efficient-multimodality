import pickle
import os.path as osp
from dmi.data.base import BaseHypnetLoader

class CANDELSLoader(BaseHypnetLoader):
    def __init__(self, tokenizer, train_args, model_name, is_instruct):
        self.max_new_tokens = 94
        self.dataset_name = 'candels'
        self.PATH = 'data/candels' 
        self.modality = 'galaxy'
        self.id_type = 'imageid'
        self.emb_name = 'emb'
        self.CAPS_PER_IMAGE = 3
        super().__init__(tokenizer, train_args, model_name, is_instruct)
        self._init_datasets()
        self._init_prefix_emb_dict()


def max_token_length():
    PATH = 'candels'
    enc_name = 'zoobot-encoder-convnext_base'
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