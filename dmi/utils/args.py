import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def default_field(obj):
    return field(default_factory=lambda: copy.deepcopy(obj))

@dataclass
class TrainArgs:
    output_dir: str
    mode: str = "train" # can be "train", "fewshot"
    device: str = "mps"
    resume_from_checkpoint: str = None
    finetune_from_checkpoint: str = None
    finetune_mm_dim: int = None
    resume_from_checkpoint_reset_steps: bool = False
    save_state: bool = True
    train_batch_size: int = 128
    subset_batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    epochs: int = None
    dataset_size: str = None
    epochs_l: List[int] = None
    dataset_size_l: List[str] = None
    warmup_steps: int = 500
    scheduler: str = "cosine_warmup"
    logging_steps: int = 50
    save_steps: int = 5000
    save_steps_l: List[int] = None
    eval_steps: int = 5000
    eval_steps_l: List[int] = None
    generate_steps: int = 5000
    generate_steps_l: List[int] = None
    eval_at_step_zero: bool = False
    generate_at_step_zero: bool = False
    seed: int = 42
    seeds: Tuple[int] = default_field(tuple((55625, 66848, 92900, 5225, 71753)))
    gradient_accumulation_steps: int = 1
    pad_to_multiple_of: int = 8
    debug: bool = False
    feed_txt_embs: bool = False
    augment_emb_space: bool = False
    subtract_mean: bool = False
    n_components: int = None

@dataclass
class MEncArgs: # Modality encoder arguments
    menc_names_or_paths: List[str]
    load_extracted_features: List[bool]
    fewshot_menc_names_or_paths: List[str] = None
    fewshot_load_extracted_features: List[bool] = None
    mm_dim: int = 768
    mm_dtype: Optional[str] = 'float32'

    
@dataclass
class LMArgs: # Language model arguments
    lm_name_or_path: str
    lm_dtype: Optional[str] = 'bfloat16'


@dataclass
class DatasetArgs: # 
    dataset_names_or_paths: List[str]
    fewshot_dataset_names_or_paths: List[str] = None


@dataclass
class ProjectorArgs:
    proj_name_or_path: str = None
    proj_arch: str = "mlp"
    proj_act: str = "quick_gelu"
    proj_n_layers: int = 2
    proj_dropout: float = 0.1
    proj_prune: int = None

@dataclass
class HypnetArgs:
    hn_name_or_path: str = "hypnet_1"
    hn_arch: str = "transformer"
    hn_n_layers: int = 1
    hn_n_heads: int = 1
    hn_hypnet_dim: int = 768 # current assumption: hypnet_dim == mm_dim
    hn_rank: int = 32
    hn_alpha: int = 32
    hn_predict_bias: bool = True
    hn_principled_init: bool = False
    hn_n_proj_layers: int = None  # set in train.py
    hn_use_pos_encs: bool = False

@dataclass
class LoraArgs:
    lora_name_or_path: str = "lora_1"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_n_proj_layers: int = None  # set in train.py

@dataclass
class FewshotArgs:
    finetune_generated_projector: bool
    fewshot_learning_rate: float = 1e-4
    fewshot_weight_decay: float = 5e-6
    fewshot_dataset_sizes: List[str] = None
    fewshot_epochs: List[int] = None
    fewshot_n_adapters: str = "multiple" # can be "one" or "multiple"
    fewshot_n_tokens: int = None


def setup_args(self, prefix, args):
    for key in dir(args):
        if key.startswith(prefix):
            key_wo_prefix = key[len(prefix):]
            setattr(self, key_wo_prefix, getattr(args, key))