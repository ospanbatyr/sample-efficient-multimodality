import os
import json
import torch
import wandb
import random
import logging
from glob import glob
import os.path as osp
from typing import List
from itertools import islice
from filelock import FileLock
from dmi.data.base import BaseLoader
from dmi.utils.args import TrainArgs
from dmi.utils.model_utils import EmbeddingManager

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LRScheduler, train_loaders: List[torch.utils.data.DataLoader], eval_loaders: List[torch.utils.data.DataLoader], emb_mgrs: List[EmbeddingManager], loader_mgrs: List[BaseLoader], device: torch.device, train_args: TrainArgs):
        self.TRAINER_TYPE = None
        self.SAVE_TYPE = None
        self.SAVE_MODEL = None
        self.WANDB_MODEL = None
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loaders = train_loaders
        self.eval_loaders = eval_loaders
        self.emb_mgrs = emb_mgrs
        self.loader_mgrs = loader_mgrs
        self.device = device
        self.train_args = train_args
        
    def _prepare_batch(self, batch, task='train'):
        assert task in ['train', 'eval'], 'task should be either train or eval'
        if task == 'train':
            input_ids, attention_masks, labels, mm_data = batch
        elif task == 'eval':
            input_ids, attention_masks, labels, mm_data, ids = batch

        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        labels = labels.to(self.device)

        if task == 'train':
            return input_ids, attention_masks, labels, mm_data
        elif task == 'eval':
            return input_ids, attention_masks, labels, mm_data, ids
        
    def _compute_losses(self, input_ids, attention_masks, labels, mm_embs):
        loss = self.model(mm_embs, input_ids, attention_masks, labels)
        return loss
    
    def _get_emb_mgr(self, idx):
        return self.emb_mgrs[idx]

    def _get_embeddings(self, mm_data, emb_mgr):
        return emb_mgr.get_embeddings(mm_data)
    
    # _train must be implemented by the subclass
    def _train(self, start_step: int):
        raise NotImplementedError("Trainers must implement this method")

    def train(self):
        start_step = self.ckpt_state['step_idx'] if self.train_args.resume_from_checkpoint else 0
        
        if self.train_args.resume_from_checkpoint:
            logging.info(f"Resuming training from step {start_step}")
            self.lr_scheduler.step(start_step)

        wandb.watch(self.WANDB_MODEL, log="gradients", log_freq=10)
        self._train(start_step)

    def _prepare_train_iterators(self, start_step: int):
        train_iterators = [iter(loader) for loader in self.train_loaders]
        total_steps = sum(len(loader) for loader in self.train_loaders)
        weights = [len(loader)/total_steps for loader in self.train_loaders]
        list_loaders = list(range(len(self.train_loaders)))

        if start_step > 0:
            logging.info(f"Resuming training from step {start_step}")
            iterator_idxs = [random.choices(population=list_loaders, weights=weights, k=1)[0] for _ in range(start_step)]
            iterator_idxs_counter = [iterator_idxs.count(i) for i in range(len(self.train_loaders))]
            for i, c in enumerate(iterator_idxs_counter):
                if c > 0:
                    train_iterators[i] = next(islice(train_iterators[i], c, c), None)
        
        return train_iterators, total_steps, weights, list_loaders

    def _log_save_test_results(self, emb_mgrs, test_metrics, test_gts, test_preds, test_ids, wandb_step_idx):
        for mgr_idx in range(len(emb_mgrs)):
            emb_mgr_name = emb_mgrs[mgr_idx].model_name_or_path.split('/')[-1]
            metrics = test_metrics[emb_mgr_name]

            for metric_name, metric in metrics.items():
                wandb.log({f"test {metric_name} - {emb_mgr_name} ": metric})

            logging.info(f"Step: {wandb_step_idx} Mgr: {emb_mgr_name} Metrics: {metrics}")

        results = dict(metrics=test_metrics, gts=test_gts, preds=test_preds, ids=test_ids)
        with open(f"../outputs/{self.TRAINER_TYPE}:{self.model.name}-results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def _get_batch(self, loaders, iterator, iterators, iterator_idx):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders[iterator_idx])
            iterators[iterator_idx] = iterator
            batch = next(iterator)

        return batch
    
    def _grad_acc_condition(self, wandb_step_idx, total_steps):
        return (wandb_step_idx == total_steps - 1) or ((wandb_step_idx + 1) % self.train_args.gradient_accumulation_steps == 0)

    def _log_train_loss(self, wandb_step_idx, total_steps, accumulated_loss):
        if (wandb_step_idx + 1) % self.train_args.logging_steps == 0 and wandb_step_idx > 0:
            wandb.log({"train_loss": accumulated_loss, "lr": self.optimizer.param_groups[0]['lr']}, step=wandb_step_idx)
            logging.info(f"Step: {wandb_step_idx}/{total_steps} Train Loss: {accumulated_loss:.3f}")

    def _log_eval_loss(self, step_idx, eval_losses, eval_loss_per_mgr, total_steps, emb_idx):
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            avg_eval_loss_per_mgr = sum(eval_loss_per_mgr) / len(eval_loss_per_mgr)
            emb_mgr_name = self.emb_mgrs[emb_idx].model_name_or_path.split('/')[-1]
            logging.info(f"Evaluating iterator {emb_mgr_name}: {step_idx}/{total_steps}, Avg Loss per Mgr: {avg_eval_loss_per_mgr:.3f}, Avg Loss: {avg_eval_loss:.3f} ")
            return avg_eval_loss

    def _eval_condition(self, wandb_step_idx, total_steps):
        if self.train_args.eval_steps_l is None:
            return  (wandb_step_idx == total_steps - 1) or ((wandb_step_idx + 1) % self.train_args.eval_steps == 0 and (wandb_step_idx > 0 or self.train_args.eval_at_step_zero))
        else:
            equal_flag = False
            for step in self.train_args.eval_steps_l:
                if wandb_step_idx + 1 == step:
                    equal_flag = True
                    break
            return equal_flag or (wandb_step_idx == total_steps - 1)
        
    def _calculate_eval_loss(self, wandb_step_idx):
        eval_loss = self.evaluate()
        self.model.train()
        wandb.log({"eval_loss": eval_loss}, step=wandb_step_idx)
        logging.info(f"Step: {wandb_step_idx} Eval Loss: {eval_loss:.3f}")
        return eval_loss
    
    def _generate_condition(self, wandb_step_idx, total_steps):
        if self.train_args.generate_steps_l is None:
            return (wandb_step_idx == total_steps - 1) or ((wandb_step_idx + 1) % self.train_args.generate_steps == 0 and (wandb_step_idx > 0 or self.train_args.generate_at_step_zero))
        else:
            equal_flag = False
            for step in self.train_args.generate_steps_l:
                if wandb_step_idx + 1 == step:
                    equal_flag = True
                    break
            return equal_flag or (wandb_step_idx == total_steps - 1) 
        
    def _save_condition(self, wandb_step_idx, total_steps):
        if self.train_args.save_steps_l is None:
            return (wandb_step_idx == total_steps - 1) or ((wandb_step_idx + 1) % self.train_args.save_steps == 0 and wandb_step_idx > 0)
        else:
            equal_flag = False
            for step in self.train_args.save_steps_l:
                if wandb_step_idx + 1 == step:
                    equal_flag = True
                    break

            return equal_flag or (wandb_step_idx == total_steps - 1)
    
    def _log_generate_metrics(self, wandb_step_idx, all_metrics, all_gts, all_preds):
        for mgr_idx in range(len(self.emb_mgrs)):
            emb_mgr_name = self.emb_mgrs[mgr_idx].model_name_or_path.split('/')[-1]

            metrics = all_metrics[emb_mgr_name]
            gts = all_gts[emb_mgr_name]
            preds = all_preds[emb_mgr_name]

            for metric_name, metric in metrics.items():
                wandb.log({f"{metric_name} - {emb_mgr_name} ": metric})

            logging.info(f"Step: {wandb_step_idx} Mgr: {emb_mgr_name} Metrics: {metrics}")

            columns = ["Expected", "Prediction"]
            data = [[gts[i], preds[i]] for i in random.sample(range(len(gts)), min(10, len(gts)))]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({f"{emb_mgr_name} - Step {wandb_step_idx}": table})

    def _prepare_generate_text(self, batch, loader_mgr, ids, gts):
        input_ids, _, _, mm_data, cur_ids = self._prepare_batch(batch, task='eval')
        cur_gts = loader_mgr.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        ids.extend(cur_ids)
        
        # works for Llama 3.x models, not guaranteed for other LMS
        if loader_mgr.is_instruct:
            cur_gts = [gt.split('assistant\n\n\n')[-1].strip() for gt in cur_gts]
        gts.extend(cur_gts)

        # if prefix is an attribute of loader_mgr
        prefix = loader_mgr.PREFIX if hasattr(loader_mgr, 'PREFIX') else loader_mgr.prefixes[0]

        if loader_mgr.is_instruct:
            prefix = loader_mgr.tokenizer.apply_chat_template([{"role": "user", "content": prefix}], return_tensors='pt', add_generation_prompt=True).to(self.device)
            prefix = prefix.expand(mm_data.shape[0], -1)
        else:
            prefix = None

        return mm_data, prefix 

    def clear_checkpoints(self):
        for file in glob(f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-step*.pt"):
            os.remove(file)
        
        for file in glob(f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-best.pt"):
            os.remove(file)
    
    def save_checkpoint(self, step_idx: int, metric: float, metric_name: str):
        os.makedirs("checkpoints", exist_ok=True)
        best_ckpt_name = f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-best.pt"

        if osp.exists(best_ckpt_name):
            checkpoint = torch.load(best_ckpt_name)
            old_metric = checkpoint[metric_name]
        else:
            old_metric = float('-inf')

        logging.info("Removing previous step checkpoints")
        for file in glob(f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-step*.pt"):
            os.remove(file)

        save_state = {
            'step_idx': step_idx,
            f'{self.SAVE_TYPE}_state_dict': self.SAVE_MODEL.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            metric_name: metric,       
        }

        if metric > old_metric:
            logging.info(f"Saving best checkpoint at step {step_idx}")
            torch.save(save_state, best_ckpt_name)

        # torch.save(save_state, f"checkpoints/{self.model.name}-checkpoint-{self.SAVE_TYPE}-step{step_idx}.pt")

    def load_checkpoint(self, resume_from_checkpoint: str):
        checkpoint = torch.load(resume_from_checkpoint)
        self.SAVE_MODEL.load_state_dict(checkpoint[f'{self.SAVE_TYPE}_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_idx = checkpoint['step_idx']
        return dict(step_idx=step_idx)

    def load_model_checkpoint(self, resume_from_checkpoint: str):
        checkpoint = torch.load(resume_from_checkpoint)
        self.SAVE_MODEL.load_state_dict(checkpoint[f'{self.SAVE_TYPE}_state_dict'])
        loss = checkpoint['loss']
        step_idx = checkpoint['step_idx']
        return dict(step_idx=step_idx, loss=loss)
    

def average_seed_results(seeds, name, dataset_size, data_args, train_type, field):
    results = list()
    for seed in seeds:
        cur_name = f"{train_type}:{name}-dsz{dataset_size}-seed{seed}"
        with open(f"../outputs/{cur_name}-results.json", 'r') as f:
            results.append(json.load(f))
    
    avg_metrics = dict()
    enc_names = list(results[0]['metrics'].keys())
    for enc_name in enc_names:
        avg_metrics[enc_name] = dict()
        for metric in results[0]['metrics'][enc_name].keys():
            avg_metrics[enc_name][metric] = sum(result['metrics'][enc_name][metric] for result in results) / len(results)

    results_file = f"../outputs/{getattr(data_args, field)[0]}-results.json"

    lock = FileLock(results_file + '.lock')
    with lock:
        results_dict = dict()
        if osp.exists(results_file):
            with open(results_file, 'r') as f:
                results_dict = json.load(f)

        results_dict[f'{train_type}:{name}-dsz{dataset_size}'] = avg_metrics

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)