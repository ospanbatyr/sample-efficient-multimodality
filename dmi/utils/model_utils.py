import torch
import wandb
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmi.model import MODEL_CLASSES, PROCESSOR_CLASSES, EMBEDDING_NAMES, F_POST_PROCESSORS, MODEL_MODALITIES, LLMS_CHATTEMPLATES


def build_tokenizer(lm_args):
    tokenizer = AutoTokenizer.from_pretrained(lm_args.lm_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if lm_args.lm_name_or_path in LLMS_CHATTEMPLATES:
        tokenizer.chat_template = LLMS_CHATTEMPLATES[lm_args.lm_name_or_path]

    return tokenizer

def build_lm(lm_args, device):
    lm_dtype = getattr(torch, lm_args.lm_dtype)
    lm = AutoModelForCausalLM.from_pretrained(lm_args.lm_name_or_path, torch_dtype=lm_dtype, device_map=device)
    return lm

class EmbeddingManager:
    def __init__(self, model_name_or_path, load_extracted_features, dtype, device, menc_args, train_args):
        self.device = device
        self.menc_args = menc_args
        self.train_args = train_args
        self.model_name_or_path = model_name_or_path
        self.load_extracted_features = load_extracted_features

        self.emb_name = EMBEDDING_NAMES[self.model_name_or_path]
        self.f_post_proc = F_POST_PROCESSORS[self.model_name_or_path]
        self.modality = MODEL_MODALITIES[self.model_name_or_path]
        
        if self.load_extracted_features:
            self.model = None
            self.processor = None
        else:
            self.model = MODEL_CLASSES[self.model_name_or_path].from_pretrained(self.model_name_or_path, torch_dtype=dtype).to(self.device)
            self.processor = PROCESSOR_CLASSES[self.model_name_or_path].from_pretrained(self.model_name_or_path)
            self._post_init()

    def _post_init(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_embeddings(self, inputs):
        if self.load_extracted_features:
            if self.train_args.feed_txt_embs and (isinstance(inputs, list) or isinstance(inputs, tuple)):
                embs, text_embs, prefix_emb = inputs
                embs = embs.to(self.device)
                text_embs = text_embs.to(self.device)
                prefix_emb = prefix_emb.to(self.device)
                text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)
                prefix_emb = prefix_emb / prefix_emb.norm(dim=1, keepdim=True)
            else:
                embs = inputs.to(self.device)

            embs = embs / embs.norm(dim=1, keepdim=True)

            new_embs = (embs, text_embs, prefix_emb) if self.train_args.feed_txt_embs and (isinstance(inputs, list) or isinstance(inputs, tuple)) else embs
            return new_embs
        else:
            model_inputs = self.processor(**inputs, return_tensors="pt").to(self.device)
            model_inputs = self.f_post_proc(model_inputs)

            with torch.no_grad():
                embs = self.model(**model_inputs)[self.emb_name]
        
            embs = embs / embs.norm(dim=1, keepdim=True)
        
        return embs

def build_embedding_managers(train_args, menc_args, device) -> List[EmbeddingManager]:
    emb_mgrs = []
    for model_name_or_path, load_extracted_features in zip(menc_args.menc_names_or_paths, menc_args.load_extracted_features):
        emb_mgr = EmbeddingManager(model_name_or_path, load_extracted_features, menc_args.mm_dtype, device, menc_args, train_args)
        emb_mgrs.append(emb_mgr)

    return emb_mgrs

def build_fewshot_embedding_managers(train_args, menc_args, device) -> List[EmbeddingManager]:
    emb_mgrs = []
    for model_name_or_path, load_extracted_features in zip(menc_args.fewshot_menc_names_or_paths, menc_args.fewshot_load_extracted_features):
        emb_mgr = EmbeddingManager(model_name_or_path, load_extracted_features, menc_args.mm_dtype, device, menc_args, train_args)
        emb_mgrs.append(emb_mgr)

    return emb_mgrs

def init_wandb(name, project, *args):
    run = wandb.init(project=project, name=name)

    run.log_code()
    for arg in args:
        run.config.update(arg)