import os
import sys
import copy
import torch
import wandb
import random
import logging
from tqdm import tqdm
import os.path as osp
from typing import List
import torch.optim as optim
from dmi.data import NAMES_LOADERS
from dmi.model.lora import LoraWrapper
from dmi.model import LLMS_CHATTEMPLATES
from dmi.model.mmmodel import LoraMMModel
from dmi.utils.eval_utils import calc_metrics
from transformers import set_seed, HfArgumentParser
from dmi.train import BaseTrainer, average_seed_results
from dmi.utils.args import DatasetArgs, LMArgs, MEncArgs, ProjectorArgs, TrainArgs, LoraArgs
from dmi.utils.model_utils import EmbeddingManager, build_tokenizer, build_lm, build_embedding_managers, init_wandb
from dmi.utils.scheduler import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_placeholder_schedule


class LoraTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler, train_loaders, eval_loaders, test_loaders, emb_mgrs: List[EmbeddingManager], loader_mgrs, device, train_args):
        super().__init__(model, optimizer, lr_scheduler, train_loaders, eval_loaders, emb_mgrs, loader_mgrs, device, train_args)
        self.TRAINER_TYPE = 'lora'
        self.SAVE_TYPE = 'lora_model'
        self.SAVE_MODEL = self.model.lora_model
        self.WANDB_MODEL = self.SAVE_MODEL
        self.test_loaders = test_loaders

        if train_args.resume_from_checkpoint:
            self.load_checkpoint(train_args.resume_from_checkpoint)

    def _train(self, start_step: int) -> float:
        self.model.train()
        train_losses = []

        train_iterators, total_steps, weights, list_loaders = self._prepare_train_iterators(start_step)

        wandb_step_idx = 0
        accumulated_loss = 0  # Added for gradient accumulation
        cur_bleu = float('-inf')
        
        for step_idx in range(start_step, total_steps):
            if step_idx % self.train_args.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
                accumulated_loss = 0
            
            wandb_step_idx = step_idx

            iterator_idx = random.choices(population=list_loaders, weights=weights, k=1)[0]
            iterator = train_iterators[iterator_idx]
            emb_mgr = self._get_emb_mgr(iterator_idx)

            batch = self._get_batch(self.train_loaders, iterator, train_iterators, iterator_idx)

            input_ids, attention_masks, labels, mm_data = self._prepare_batch(batch)
            mm_embs = self._get_embeddings(mm_data, emb_mgr)

            loss = self._compute_losses(input_ids, attention_masks, labels, mm_embs) / self.train_args.gradient_accumulation_steps

            loss.backward()
            accumulated_loss += loss.item()

            if self._grad_acc_condition(wandb_step_idx, total_steps):
                torch.nn.utils.clip_grad_norm_(self.model.lora_model.parameters(), self.train_args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step(wandb_step_idx)
                train_losses.append(accumulated_loss)

                self._log_train_loss(wandb_step_idx, total_steps, accumulated_loss)
                if self._eval_condition(wandb_step_idx, total_steps):
                    self._calculate_eval_loss(wandb_step_idx)

                if self._generate_condition(wandb_step_idx, total_steps):
                    all_metrics, all_gts, all_preds, all_ids = self.generate(mode='eval')
                    self.model.train()

                    assert len(all_metrics) == 1, "Currently only one embedding manager is supported for generation"
                    metric_names = list(all_metrics[list(all_metrics.keys())[0]].keys())
                    comp_metric = 'coco_cider' if 'coco_cider' in metric_names else 'bleu'
                    cur_metric = sum(all_metrics[k][comp_metric] for k in all_metrics) / len(all_metrics)

                    self._log_generate_metrics(wandb_step_idx, all_metrics, all_gts, all_preds)

                if self._save_condition(wandb_step_idx, total_steps):
                    self.save_checkpoint(wandb_step_idx, cur_metric, comp_metric)

        self.load_checkpoint(f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-best.pt")

        test_metrics, test_gts, test_preds, test_ids = self.generate(mode='test')
        self._log_save_test_results(self.emb_mgrs, test_metrics, test_gts, test_preds, test_ids, wandb_step_idx)

    def evaluate(self):
        self.model.eval()
        eval_losses = []

        eval_iterators = [iter(loader) for loader in self.eval_loaders]

        for emb_idx in range(len(self.emb_mgrs)):
            emb_mgr = self.emb_mgrs[emb_idx]
            iterator = eval_iterators[emb_idx]
            total_steps = len(self.eval_loaders[emb_idx])
            eval_loss_per_mgr = []

            for step_idx, batch in enumerate(iterator):
                input_ids, attention_masks, labels, mm_data, _ = self._prepare_batch(batch, task='eval')

                with torch.no_grad():
                    mm_embs = self._get_embeddings(mm_data, emb_mgr)
                    loss = self._compute_losses(input_ids, attention_masks, labels, mm_embs)

                eval_losses.append(loss.item())
                eval_loss_per_mgr.append(loss.item())

                if step_idx % self.train_args.logging_steps == 0 and step_idx > 0:
                    avg_eval_loss_per_mgr = sum(eval_loss_per_mgr) / len(eval_loss_per_mgr)
                    logging.info(f"Evaluating {step_idx}/{total_steps}: Avg Loss: {avg_eval_loss_per_mgr:.3f}")

            self._log_eval_loss(step_idx, eval_losses, eval_loss_per_mgr, total_steps, emb_idx)

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        return avg_eval_loss

    def generate(self, mode='eval'):
        assert mode in ['eval', 'test'], 'mode should be either eval or test'
        self.model.eval()
        all_metrics, all_gts, all_preds, all_ids = dict(), dict(), dict(), dict()

        cur_loaders = self.eval_loaders if mode == 'eval' else self.test_loaders
        cur_iterators = [iter(loader) for loader in cur_loaders]

        for emb_idx in range(len(self.emb_mgrs)):
            iterator = cur_iterators[emb_idx]
            loader_mgr = self.loader_mgrs[emb_idx]
            total_steps = len(cur_loaders[emb_idx])
            emb_mgr_name = self.emb_mgrs[emb_idx].model_name_or_path.split('/')[-1]

            gts, preds, ids = list(), list(), list()

            for step_idx, batch in tqdm(enumerate(iterator), miniters=self.train_args.logging_steps, total=total_steps):
                mm_data, prefix = self._prepare_generate_text(batch, loader_mgr, ids, gts)

                with torch.no_grad():
                    mm_embs = self._get_embeddings(mm_data, self.emb_mgrs[emb_idx])

                outputs = self.model.generate(mm_embs, loader_mgr.max_new_tokens, prefix=prefix)
                cur_preds = loader_mgr.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds.extend(cur_preds)
    
            all_gts[emb_mgr_name] = gts
            all_preds[emb_mgr_name] = preds
            all_ids[emb_mgr_name] = ids

            metrics = calc_metrics(preds, ids, loader_mgr.dataset_name, self.model.name, mode)
            all_metrics[emb_mgr_name] = metrics

        return all_metrics, all_gts, all_preds, all_ids

def args_post_init(train_args: TrainArgs, menc_args: MEncArgs, lora_args: LoraArgs, projector_args: ProjectorArgs):
    lora_args.lora_n_proj_layers = projector_args.proj_n_layers
    if train_args.finetune_mm_dim is not None:
        if menc_args.mm_dim < train_args.finetune_mm_dim:
            projector_args.proj_prune = menc_args.mm_dim
        elif menc_args.mm_dim > train_args.finetune_mm_dim:
            train_args.n_components = train_args.finetune_mm_dim
            menc_args.mm_dim = train_args.finetune_mm_dim

def main(name, data_args, lora_args, lm_args, menc_args, projector_args, train_args):
    device = train_args.device
    is_instruct = lm_args.lm_name_or_path in LLMS_CHATTEMPLATES

    if train_args.debug:
        train_args.train_batch_size //= 32
        train_args.subset_batch_size //= 32
        train_args.eval_batch_size //= 32
        train_args.eval_steps = 1
        train_args.generate_steps = 4
        train_args.logging_steps = 1
        train_args.save_steps = 2

        os.environ["WANDB_MODE"] = "disabled"

    args_post_init(train_args, menc_args, lora_args, projector_args)

    logging.info("Building tokenizer")
    tokenizer = build_tokenizer(lm_args)

    logging.info("Building language model")
    lm = build_lm(lm_args, device)

    logging.info("Building embedding managers")
    emb_mgrs = build_embedding_managers(train_args, menc_args, device)

    lm_emb_dim = lm.config.hidden_size
    mm_emb_dim = menc_args.mm_dim

    logging.info("Building lora model")
    lora_model = LoraWrapper(
        lora_args=lora_args,
        proj_args=projector_args,
        lm_emb_dim=lm_emb_dim, 
        mm_emb_dim=mm_emb_dim, 
        device=device
    ).to(device)

    logging.info("Building model")
    model = LoraMMModel(
        llm=lm, 
        lora_model=lora_model, 
        device=device, 
        mm_emb_dim=mm_emb_dim,
        name=name,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.lora_model.trainable_parameters() if p.requires_grad)}")

    optimizer = optim.AdamW(
        params=model.lora_model.trainable_parameters(), 
        lr=train_args.learning_rate, 
        weight_decay=train_args.weight_decay,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon
    )

    logging.info("Building loaders")
    loader_mgr_args_base = (tokenizer, train_args)

    model_names = [model_name.split('/')[-1] for model_name in menc_args.menc_names_or_paths]
    loader_mgrs_args = [(*loader_mgr_args_base, model_name, is_instruct) for model_name in model_names]

    loader_mgrs = []
    for dataset_name in data_args.dataset_names_or_paths:
        loader_mgrs.append(NAMES_LOADERS[dataset_name])

    loader_mgrs = [loader_mgr(*loader_mgr_args) for loader_mgr, loader_mgr_args in zip(loader_mgrs, loader_mgrs_args)]
    loaders = [loader_mgr.build_eval_and_test_loaders() for loader_mgr in loader_mgrs]
    train_loaders = [loader[0] for loader in loaders]
    eval_loaders = [loader[1] for loader in loaders]
    test_loaders = [loader[2] for loader in loaders]

    logging.info("Building schedulers")
    warmup_steps = train_args.warmup_steps
    total_steps = sum(len(loader) for loader in train_loaders)

    if train_args.scheduler == 'linear_warmup':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
        )
    elif train_args.scheduler == 'cosine_warmup':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif train_args.scheduler is None:
        lr_scheduler = get_placeholder_schedule(optimizer)
    else:
        raise ValueError("Scheduler should be either linear_warmup or cosine_warmup")

    trainer = LoraTrainer(
        model=model, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loaders=train_loaders, 
        eval_loaders=eval_loaders, 
        test_loaders=test_loaders,
        emb_mgrs=emb_mgrs,
        loader_mgrs=loader_mgrs,
        device=device, 
        train_args=train_args
    )

    logging.info("Starting training")
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    logging.info("Starting parsing")

    parser = HfArgumentParser(
        (DatasetArgs, LoraArgs, LMArgs, MEncArgs, ProjectorArgs, TrainArgs)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, lora_args, lm_args, menc_args, projector_args, train_args = parser.parse_json_file(
            json_file=osp.abspath(sys.argv[1])
        )
        name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        (
            data_args, lora_args, lm_args, menc_args, projector_args, train_args
        ) = parser.parse_args_into_dataclasses()
        name = None

    seeds = train_args.seeds
    train_args.seeds = None

    epochs_l = train_args.epochs_l
    dataset_size_l = train_args.dataset_size_l

    train_args.epochs_l = None
    train_args.dataset_size_l = None

    for epochs, dataset_size in zip(epochs_l, dataset_size_l):
        train_args.epochs = epochs
        train_args.dataset_size = dataset_size
        logging.info(f"Training for {train_args.epochs} epochs with dataset size {train_args.dataset_size}")
        train_type = 'lora'
        
        for seed in seeds:
            logging.info(f"Training for seed {seed}")
            train_args.seed = seed
            set_seed(seed)

            output_fname = f"{train_type}:{name}-dsz{dataset_size}-seed{seed}"
            if osp.exists(f"../outputs/{output_fname}-results.json"):
                logging.info(f"Skipping {output_fname} because it already exists")
                continue
            
            cur_name = f"{name}-dsz{dataset_size}-seed{seed}"
            init_wandb(cur_name, 'dmi_lora', data_args, lora_args, lm_args, menc_args, projector_args, train_args)
            main(cur_name, copy.deepcopy(data_args), copy.deepcopy(lora_args), copy.deepcopy(lm_args), copy.deepcopy(menc_args), copy.deepcopy(projector_args), copy.deepcopy(train_args))
            wandb.finish()

        if len(data_args.dataset_names_or_paths) == 1:
            average_seed_results(seeds, name, dataset_size, data_args, train_type=train_type, field='dataset_names_or_paths')