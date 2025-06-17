import os
import sys
import copy
import torch
import wandb
import random
import logging
import os.path as osp
from tqdm import tqdm
from glob import glob
from typing import List
import torch.optim as optim
from itertools import islice
from dmi.data import NAMES_LOADERS
from scipy.stats import ortho_group
from dmi.model import LLMS_CHATTEMPLATES
from dmi.model.mmmodel import HypernetMMModel
from dmi.utils.eval_utils import calc_metrics
from dmi.model.hypernet import HyperNetWrapper
from transformers import set_seed, HfArgumentParser
from dmi.train import BaseTrainer, average_seed_results
from dmi.utils.scheduler import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from dmi.utils.args import DatasetArgs, HypnetArgs, LMArgs, MEncArgs, ProjectorArgs, TrainArgs, FewshotArgs
from dmi.utils.model_utils import EmbeddingManager, build_tokenizer, build_lm, build_embedding_managers, build_fewshot_embedding_managers, init_wandb

class HypernetTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler, train_loaders, train_subset_loaders, eval_loaders, eval_subset_loaders, emb_mgrs: List[EmbeddingManager], loader_mgrs, device, train_args, fewshot_train_loaders, fewshot_train_subset_loaders, fewshot_eval_loaders, fewshot_eval_subset_loaders, fewshot_emb_mgrs, fewshot_loader_mgrs, fewshot_args, fewshot_test_loaders=None, fewshot_test_subset_loaders=None,):
        super().__init__(model, optimizer, lr_scheduler, train_loaders, eval_loaders, emb_mgrs, loader_mgrs, device, train_args)
        self.TRAINER_TYPE = 'hypernet'
        self.SAVE_TYPE = 'hypernet'
        self.SAVE_MODEL = self.model.hypernet
        self.WANDB_MODEL = self.model.hypernet.hypernet

        self.train_subset_loaders = train_subset_loaders
        self.eval_subset_loaders = eval_subset_loaders
        self.fewshot_train_loaders = fewshot_train_loaders
        self.fewshot_train_subset_loaders = fewshot_train_subset_loaders
        self.fewshot_eval_loaders = fewshot_eval_loaders
        self.fewshot_eval_subset_loaders = fewshot_eval_subset_loaders
        self.fewshot_test_loaders = fewshot_test_loaders
        self.fewshot_test_subset_loaders = fewshot_test_subset_loaders
        self.fewshot_emb_mgrs = fewshot_emb_mgrs
        self.fewshot_loader_mgrs = fewshot_loader_mgrs
        self.fewshot_args = fewshot_args

        if train_args.resume_from_checkpoint:
            self.load_checkpoint(train_args.resume_from_checkpoint)

    def _compute_losses(self, input_ids, attention_masks, labels, mm_embs, mm_subset_embs):        
        return self.model(mm_embs, mm_subset_embs, input_ids, attention_masks, labels)[0]
    
    def _get_emb_mgr(self, idx, fewshot=False):
        emb_mgr = self.fewshot_emb_mgrs[idx] if fewshot else self.emb_mgrs[idx]
        return emb_mgr

    def _get_rotation_matrix(self, mm_dim):
        return torch.FloatTensor(ortho_group.rvs(mm_dim)).to(self.device)

    def _prepare_train_iterators(self, start_step: int):
        train_iterators = [iter(loader) for loader in self.train_loaders]
        train_subset_iterators = [iter(loader) for loader in self.train_subset_loaders]
        total_steps = sum(len(loader) for loader in self.train_loaders)

        if start_step > 0:
            logging.info(f"Resuming training from step {start_step}")
            iterator_idxs = [random.randint(0, len(self.train_loaders) - 1) for _ in range(start_step)]
            iterator_idxs_counter = [iterator_idxs.count(i) for i in range(len(self.train_loaders))]
            
            for i, c in enumerate(iterator_idxs_counter):
                if c > 0:
                    train_iterators[i] = next(islice(train_iterators[i], c, c), None)
                    train_subset_iterators[i] = next(islice(train_subset_iterators[i], c, c), None)

        return train_iterators, train_subset_iterators, total_steps
    
    def _interleave_embeddings(self, mm_subset_membs, txt_embs):
        # Interleave the text and multi-modal embeddings
        mm_subset_embs = torch.stack(
            (mm_subset_membs, txt_embs), 
            dim=0
        ).transpose(0, 1).reshape(-1, *mm_subset_membs.shape[1:])

        return mm_subset_embs

    def _process_embeddings(self, mm_embs, mm_subset_embs, can_rotate):
        assert isinstance(can_rotate, bool), 'can_rotate should be a boolean'

        if can_rotate and self.train_args.augment_emb_space:
            R = self._get_rotation_matrix(mm_embs.shape[1])

        if self.train_args.feed_txt_embs:
            mm_subset_membs, txt_embs, prefix_emb = mm_subset_embs

            # Rotate the multi-modal embeddings if the flag is set
            if can_rotate and self.train_args.augment_emb_space:
                mm_embs = mm_embs @ R
                mm_subset_membs = mm_subset_membs @ R

            if self.SAVE_MODEL.projector.prune is not None:
                mm_subset_membs = torch.nn.functional.pad(mm_subset_membs, (0, self.train_args.finetune_mm_dim - self.SAVE_MODEL.projector.prune, 0, 0))

            # Interleave the text and multi-modal embeddings
            mm_subset_embs = self._interleave_embeddings(mm_subset_membs, txt_embs)

            # Concatenate the prefix embedding with the interleaved embeddings
            mm_subset_embs = torch.cat([prefix_emb, mm_subset_embs], dim=0)

        return mm_embs, mm_subset_embs

    def _train(self, start_step: int) -> float:
        self.model.train()
        train_losses = []

        train_iterators, train_subset_iterators, total_steps = self._prepare_train_iterators(start_step)
        wandb_step_idx = 0
        accumulated_loss = 0

        for step_idx in range(start_step, total_steps):
            if step_idx % self.train_args.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
                accumulated_loss = 0
            
            wandb_step_idx = step_idx

            iterator_idx = random.randint(0, len(self.train_loaders) - 1)
            iterator = train_iterators[iterator_idx]
            subset_iterator = train_subset_iterators[iterator_idx]
            emb_mgr = self._get_emb_mgr(iterator_idx)

            batch = self._get_batch(self.train_loaders, iterator, train_iterators, iterator_idx)
            input_ids, attention_masks, labels, mm_data = self._prepare_batch(batch)

            subset_mm_data = self._get_batch(self.train_subset_loaders, subset_iterator, train_subset_iterators, iterator_idx)
            
            mm_embs = self._get_embeddings(mm_data, emb_mgr)
            mm_subset_embs = self._get_embeddings(subset_mm_data, emb_mgr)

            mm_embs, mm_subset_embs = self._process_embeddings(mm_embs, mm_subset_embs, can_rotate=True)

            # Scale loss by accumulation steps
            loss = self._compute_losses(input_ids, attention_masks, labels, mm_embs, mm_subset_embs) / self.train_args.gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()

            # Only update weights after accumulating enough gradients
            if self._grad_acc_condition(wandb_step_idx, total_steps):
                torch.nn.utils.clip_grad_norm_(self.model.hypernet.parameters(), self.train_args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step(wandb_step_idx)
                train_losses.append(accumulated_loss)

                self._log_train_loss(wandb_step_idx, total_steps, accumulated_loss)

                if self._eval_condition(wandb_step_idx, total_steps):
                    eval_loss = self._calculate_eval_loss(wandb_step_idx)

                if self._generate_condition(wandb_step_idx, total_steps):
                    all_metrics, all_gts, all_preds, all_ids = self.generate(mode='eval')
                    self.model.train()
                    self._log_generate_metrics(wandb_step_idx, all_metrics, all_gts, all_preds)

                if self._save_condition(wandb_step_idx, total_steps):
                    self.save_checkpoint(wandb_step_idx, eval_loss)

        # self.fewshot_generate(train_step_idx=wandb_step_idx)

    def fewshot_generate_adapters(self, emb_idx, subset_iterator, emb_mgr):
        if self.fewshot_args.finetune_generated_projector:
            zs = []
            if self.fewshot_args.fewshot_n_adapters == "one":
                n_subsets = 1
            elif self.fewshot_args.fewshot_n_adapters == "multiple":
                n_subsets = len(self.fewshot_train_loaders[emb_idx].dataset) // self.train_args.subset_batch_size
            else:
                raise ValueError(f"Invalid fewshot_n_adapters: {self.fewshot_args.fewshot_n_adapters}")
            
            logging.info(f"Generating {n_subsets} adapters for fewshot training")

            for _ in range(n_subsets):
                initial_mm_data = next(subset_iterator)
                mm_subset_embs = self._get_embeddings(initial_mm_data, emb_mgr)

                if self.train_args.feed_txt_embs:
                    mm_subset_membs, txt_embs, prefix_emb = mm_subset_embs
                else:
                    mm_subset_membs = mm_subset_embs
                    
                if self.SAVE_MODEL.projector.prune is not None:
                    mm_subset_membs = torch.nn.functional.pad(mm_subset_membs, (0, self.train_args.finetune_mm_dim - self.SAVE_MODEL.projector.prune, 0, 0))

                if self.train_args.feed_txt_embs:
                    mm_subset_embs = self._interleave_embeddings(mm_subset_membs, txt_embs)
                    mm_subset_embs = torch.cat([prefix_emb, mm_subset_embs], dim=0)
                else:
                    mm_subset_embs = mm_subset_membs

                zs.append(mm_subset_embs)
            
            self.model.hypernet.generate_projector_from_multiple_adapters(zs)
    
    def fewshot_generate(self, train_step_idx=None):
        fewshot_train_losses = []
        fewshot_iterators = [iter(loader) for loader in self.fewshot_train_loaders]
        fewshot_subset_iterators = [iter(loader) for loader in self.fewshot_train_subset_loaders]
        fewshot_all_test_metrics, fewshot_all_test_gts, fewshot_all_test_preds, fewshot_all_test_ids = dict(), dict(), dict(), dict()

        for emb_idx in range(len(self.fewshot_emb_mgrs)):
            emb_mgr = self.fewshot_emb_mgrs[emb_idx]
            fewshot_iterator = fewshot_iterators[emb_idx]
            subset_iterator = fewshot_subset_iterators[emb_idx]
            total_steps = len(self.fewshot_train_loaders[emb_idx])
            emb_mgr_name = emb_mgr.model_name_or_path.split('/')[-1]


            self.fewshot_generate_adapters(emb_idx, subset_iterator, emb_mgr)

            self.model.train()

            self.optimizer = optim.AdamW(
                params=self.model.hypernet.trainable_parameters(),
                lr=self.fewshot_args.fewshot_learning_rate,
                weight_decay=self.fewshot_args.fewshot_weight_decay
            )

            best_metric = float('-inf')
            accumulated_loss = 0
            
            for step_idx in range(total_steps):
                if step_idx % self.train_args.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss = 0

                batch = next(fewshot_iterator)
                input_ids, attention_masks, labels, mm_data = self._prepare_batch(batch)
                subset_iterator = fewshot_subset_iterators[emb_idx]

                subset_mm_data = self._get_batch(self.fewshot_train_subset_loaders, subset_iterator, fewshot_subset_iterators, emb_idx)

                mm_embs = self._get_embeddings(mm_data, emb_mgr)
                mm_subset_embs = self._get_embeddings(subset_mm_data, emb_mgr)

                mm_embs, mm_subset_embs = self._process_embeddings(mm_embs, mm_subset_embs, can_rotate=False) # we will not rotate the embeddings here

                loss = self._compute_losses(input_ids, attention_masks, labels, mm_embs, mm_subset_embs) / self.train_args.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if self._grad_acc_condition(step_idx, total_steps):
                    torch.nn.utils.clip_grad_norm_(self.model.hypernet.parameters(), self.train_args.max_grad_norm)
                    self.optimizer.step()

                    fewshot_train_losses.append(accumulated_loss)

                    self._log_train_loss(step_idx, total_steps, accumulated_loss)

                    if self._eval_condition(step_idx, total_steps):
                        all_metrics, _, _, _ = self.generate(
                            fewshot=True,
                            fewshot_emb_mgr=emb_mgr,
                            fewshot_loader_mgr=self.fewshot_loader_mgrs[emb_idx],
                            fewshot_eval_loader=self.fewshot_eval_loaders[emb_idx],
                            fewshot_eval_subset_loader=self.fewshot_eval_subset_loaders[emb_idx],
                            mode='eval'
                        )
                        self.model.train()
                        metric_names = list(all_metrics[list(all_metrics.keys())[0]].keys())
                        comp_metric = 'coco_cider' if 'coco_cider' in metric_names else 'bleu'
                        cur_metric = sum(all_metrics[k][comp_metric] for k in all_metrics) / len(all_metrics)

                        if best_metric < cur_metric:
                            print(f"Best {comp_metric}: {best_metric} < {cur_metric}")
                            best_metric = cur_metric
                            self.save_fewshot_model_checkpoint(step_idx, cur_metric, comp_metric)

            self.load_fewshot_model_checkpoint(comp_metric)
            test_metrics, test_gts, test_preds, test_ids = self.generate(
                fewshot=True,
                fewshot_emb_mgr=emb_mgr,
                fewshot_loader_mgr=self.fewshot_loader_mgrs[emb_idx],
                fewshot_eval_loader=self.fewshot_test_loaders[emb_idx],
                fewshot_eval_subset_loader=self.fewshot_test_subset_loaders[emb_idx],
                mode='test'
            )

            self.model.train()
            fewshot_all_test_metrics[emb_mgr_name] = test_metrics[emb_mgr_name]
            fewshot_all_test_gts[emb_mgr_name] = test_gts[emb_mgr_name]
            fewshot_all_test_preds[emb_mgr_name] = test_preds[emb_mgr_name]
            fewshot_all_test_ids[emb_mgr_name] = test_ids[emb_mgr_name]

        self._log_save_test_results(self.fewshot_emb_mgrs, fewshot_all_test_metrics, fewshot_all_test_gts, fewshot_all_test_preds, fewshot_all_test_ids, train_step_idx)

        del self.model.hypernet.generated_projector
        self.model.hypernet.generated_projector = None


    def _generate_condition(self, wandb_step_idx, total_steps):
        if self.train_args.generate_steps_l is None:
            return ((wandb_step_idx + 1) % self.train_args.generate_steps == 0 and (wandb_step_idx > 0 or self.train_args.generate_at_step_zero))
        else:
            equal_flag = False
            for step in self.train_args.generate_steps_l:
                if wandb_step_idx + 1 == step:
                    equal_flag = True
                    break
            return equal_flag or (wandb_step_idx == total_steps - 1) 


    def evaluate(self, fewshot=False, fewshot_emb_mgr=None, fewshot_eval_loader=None, fewshot_eval_subset_loader=None):
        self.model.eval()
        eval_losses = []

        if fewshot:
            assert fewshot_emb_mgr is not None and fewshot_eval_loader is not None and fewshot_eval_subset_loader is not None, \
                "fewshot_emb_mgr, fewshot_eval_loader, and fewshot_eval_subset_loader should be provided"
            emb_mgrs, cur_loaders, cur_subset_loaders = [fewshot_emb_mgr], [fewshot_eval_loader], [fewshot_eval_subset_loader]
        else:
            emb_mgrs, cur_loaders, cur_subset_loaders = self.emb_mgrs, self.eval_loaders, self.eval_subset_loaders

        eval_iterators = [iter(loader) for loader in cur_loaders]
        eval_subset_iterators = [iter(loader) for loader in cur_subset_loaders]

        for emb_idx in range(len(emb_mgrs)):
            emb_mgr = emb_mgrs[emb_idx]
            eval_iterator = eval_iterators[emb_idx]
            total_steps = len(cur_loaders[emb_idx])
            eval_loss_per_mgr = []

            for step_idx, batch in enumerate(eval_iterator):
                input_ids, attention_masks, labels, mm_data, _ = self._prepare_batch(batch, task='eval')

                eval_subset_iterator = eval_subset_iterators[emb_idx]
                subset_mm_data = self._get_batch(cur_subset_loaders, eval_subset_iterator, eval_subset_iterators, emb_idx)
                
                with torch.no_grad():
                    mm_embs = self._get_embeddings(mm_data, emb_mgr)
                    mm_subset_embs = self._get_embeddings(subset_mm_data, emb_mgr)

                    mm_embs, mm_subset_embs = self._process_embeddings(mm_embs, mm_subset_embs, can_rotate=False)
                    loss = self._compute_losses(input_ids, attention_masks, labels, mm_embs, mm_subset_embs)

                eval_losses.append(loss.item())
                eval_loss_per_mgr.append(loss.item())

                if step_idx % self.train_args.logging_steps == 0 and step_idx > 0:
                    avg_eval_loss_per_mgr = sum(eval_loss_per_mgr) / len(eval_loss_per_mgr)
                    logging.info(f"Evaluating {step_idx}/{total_steps}: Avg Loss: {avg_eval_loss_per_mgr:.3f}")

            avg_eval_loss = self._log_eval_loss(step_idx, eval_losses, eval_loss_per_mgr, total_steps, emb_idx)

        return avg_eval_loss

    def generate(self, fewshot=False, fewshot_emb_mgr=None, fewshot_loader_mgr=None, fewshot_eval_loader=None, fewshot_eval_subset_loader=None, mode='eval'):
        assert mode in ['eval', 'test'], 'mode should be either eval or test'
        self.model.eval()
        all_metrics, all_gts, all_preds, all_ids = dict(), dict(), dict(), dict()

        if fewshot:
            assert all(var is not None for var in (fewshot_emb_mgr, fewshot_loader_mgr, fewshot_eval_loader, fewshot_eval_subset_loader)), \
                "fewshot_emb_mgr, fewshot_loader_mgr, fewshot_eval_loader, and fewshot_eval_subset_loader should be provided"
            emb_mgrs, loader_mgrs, cur_loaders, cur_subset_loaders = [fewshot_emb_mgr], [fewshot_loader_mgr], [fewshot_eval_loader], [fewshot_eval_subset_loader]
        else:
            emb_mgrs, loader_mgrs, cur_loaders, cur_subset_loaders = self.emb_mgrs, self.loader_mgrs, self.eval_loaders, self.eval_subset_loaders

        iterators = [iter(loader) for loader in cur_loaders]
        subset_iterators = [iter(loader) for loader in cur_subset_loaders]

        for emb_idx in range(len(emb_mgrs)):
            emb_mgr = emb_mgrs[emb_idx]
            iterator = iterators[emb_idx]
            loader_mgr = loader_mgrs[emb_idx]
            total_steps = len(cur_loaders[emb_idx])
            emb_mgr_name = emb_mgrs[emb_idx].model_name_or_path.split('/')[-1]

            gts, preds, ids = list(), list(), list()

            for step_idx, batch in tqdm(enumerate(iterator), miniters=self.train_args.logging_steps, total=total_steps):
                mm_data, prefix = self._prepare_generate_text(batch, loader_mgr, ids, gts)

                subset_iterator = subset_iterators[emb_idx]
                subset_mm_data = self._get_batch(cur_subset_loaders, subset_iterator, subset_iterators, emb_idx)
                    
                emb_mgr = self._get_emb_mgr(emb_idx, fewshot)

                with torch.no_grad():
                    mm_embs = self._get_embeddings(mm_data, emb_mgr)
                    mm_subset_embs = self._get_embeddings(subset_mm_data, emb_mgr)
                    mm_embs, mm_subset_embs = self._process_embeddings(mm_embs, mm_subset_embs, can_rotate=False)

                outputs = self.model.generate(mm_embs, mm_subset_embs, loader_mgr.max_new_tokens, prefix=prefix)
                cur_preds = loader_mgr.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds.extend(cur_preds)
        
            all_gts[emb_mgr_name] = gts
            all_preds[emb_mgr_name] = preds
            all_ids[emb_mgr_name] = ids

            metrics = calc_metrics(preds, ids, loader_mgr.dataset_name, self.model.name, mode)
            all_metrics[emb_mgr_name] = metrics

        return all_metrics, all_gts, all_preds, all_ids
    
    def save_fewshot_model_checkpoint(self, step_idx: int, metric: float, metric_name: str):
        os.makedirs("checkpoints", exist_ok=True)
        best_ckpt_name = f"checkpoints/{self.model.name}-checkpoint-fewshot-best.pt"

        save_state = {
            'step_idx': step_idx,
            'hypernet_state_dict': self.model.hypernet.state_dict(),
            metric_name: metric,
        }

        logging.info(f"Saving best fewshot checkpoint at step {step_idx}")
        torch.save(save_state, best_ckpt_name)

    def load_checkpoint(self, resume_from_checkpoint: str):
        checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
        if self.SAVE_MODEL.projector.prune is not None:
            for layer in checkpoint[f'{self.SAVE_TYPE}_state_dict']:
                if 'net.0.weight' in layer:
                    checkpoint[f'{self.SAVE_TYPE}_state_dict'][layer] = checkpoint[f'{self.SAVE_TYPE}_state_dict'][layer][:, :self.SAVE_MODEL.projector.prune]

        self.SAVE_MODEL.load_state_dict(checkpoint[f'{self.SAVE_TYPE}_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_idx = checkpoint['step_idx']
        return dict(step_idx=step_idx)
    
    def load_fewshot_model_checkpoint(self, metric_name: str):
        best_ckpt_name = f"checkpoints/{self.model.name}-checkpoint-fewshot-best.pt"
        checkpoint = torch.load(best_ckpt_name)
        self.model.hypernet.load_state_dict(checkpoint['hypernet_state_dict'])
        metric = checkpoint[metric_name]
        step_idx = checkpoint['step_idx']
        return dict(step_idx=step_idx, metric=metric)

    def save_checkpoint(self, step_idx: int, loss: float):
        os.makedirs("checkpoints", exist_ok=True)
        best_ckpt_name = f"checkpoints/{self.model.name}-checkpoint-hypernet-best.pt"

        if osp.exists(best_ckpt_name):
            checkpoint = torch.load(best_ckpt_name)
            old_loss = checkpoint['loss']
        else:
            old_loss = float('inf')

        logging.info("Removing previous step checkpoints")
        for file in glob(f"checkpoints/{self.model.name}-checkpoint-hypernet-step*.pt"):
            os.remove(file)

        save_state = {
            'step_idx': step_idx,
            'hypernet_state_dict': self.model.hypernet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        if loss < old_loss:
            logging.info(f"Saving best checkpoint at step {step_idx}")
            torch.save(save_state, best_ckpt_name)

        torch.save(save_state, f"checkpoints/{self.model.name}-checkpoint-hypernet-step{step_idx}.pt")


def args_post_init(hn_args: HypnetArgs, projector_args: ProjectorArgs):
    hn_args.hn_n_proj_layers = projector_args.proj_n_layers
    if train_args.finetune_mm_dim is not None:
        if menc_args.mm_dim < train_args.finetune_mm_dim:
            projector_args.proj_prune = menc_args.mm_dim
        elif menc_args.mm_dim > train_args.finetune_mm_dim:
            train_args.n_components = train_args.finetune_mm_dim
            menc_args.mm_dim = train_args.finetune_mm_dim


def main(name: str, train_args: TrainArgs, hn_args: HypnetArgs, projector_args: ProjectorArgs, data_args: DatasetArgs, menc_args: MEncArgs, lm_args: LMArgs, fewshot_args: FewshotArgs):
    device = train_args.device
    is_instruct = lm_args.lm_name_or_path in LLMS_CHATTEMPLATES
    assert train_args.mode in ['train', 'fewshot'], "mode should be either train or fewshot"

    if train_args.debug:
        train_args.train_batch_size = 4
        train_args.subset_batch_size = 128
        train_args.eval_batch_size = 4
        train_args.eval_steps = 1
        train_args.generate_steps = 4
        train_args.logging_steps = 1
        train_args.save_steps = 2

        os.environ["WANDB_MODE"] = "disabled"

    logging.info("Building tokenizer")
    tokenizer = build_tokenizer(lm_args)

    logging.info("Building language model")
    lm = build_lm(lm_args, device)

    logging.info("Building embedding managers")
    emb_mgrs = build_embedding_managers(train_args, menc_args, device)
    fewshot_emb_mgrs = build_fewshot_embedding_managers(train_args, menc_args, device)

    lm_emb_dim = lm.config.hidden_size
    mm_emb_dim = menc_args.mm_dim

    logging.info("Building hypernet")
    hypernet = HyperNetWrapper(
        hn_args=hn_args,
        proj_args=projector_args,
        lm_emb_dim=lm_emb_dim, 
        mm_emb_dim=mm_emb_dim, 
        n_tokens=fewshot_args.fewshot_n_tokens if fewshot_args.fewshot_n_tokens is not None else train_args.subset_batch_size, 
        device=device,
    ).to(device)

    logging.info("Building model")
    model = HypernetMMModel(
        llm=lm, 
        hypernet=hypernet, 
        device=device, 
        mm_emb_dim=mm_emb_dim,
        name=name,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.hypernet.trainable_parameters() if p.requires_grad)}")

    optimizer = optim.AdamW(
        params=model.hypernet.trainable_parameters(), 
        lr=train_args.learning_rate, 
        weight_decay=train_args.weight_decay,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon
    )

    def build_loaders(cur_train_args):
        loader_mgr_args_base = (tokenizer, cur_train_args)
        model_names = [model_name.split('/')[-1] for model_name in menc_args.menc_names_or_paths]
        loader_mgrs_args = [(*loader_mgr_args_base, model_name, is_instruct) for model_name in model_names]

        loader_mgrs = []
        for dataset_name in data_args.dataset_names_or_paths:
            loader_mgrs.append(NAMES_LOADERS[dataset_name])

        loader_mgrs = [loader_mgr(*loader_mgr_args) for loader_mgr, loader_mgr_args in zip(loader_mgrs, loader_mgrs_args)]
        loaders = [loader_mgr.build_hypnet_loaders() for loader_mgr in loader_mgrs]

        train_loaders = [loader[0] for loader in loaders]
        train_subset_loaders = [loader[1] for loader in loaders]
        eval_loaders = [loader[2] for loader in loaders]
        eval_subset_loaders = [loader[3] for loader in loaders]
        return loader_mgrs, train_loaders, train_subset_loaders, eval_loaders, eval_subset_loaders

    logging.info("Building fewshot loaders")
    def build_fewshot_loaders(cur_train_args):
        fewshot_loader_mgr_args_base = (tokenizer, cur_train_args)
        fewshot_model_names = [model_name.split('/')[-1] for model_name in menc_args.fewshot_menc_names_or_paths]
        fewshot_loader_mgrs_args = [(*fewshot_loader_mgr_args_base, model_name, is_instruct) for model_name in fewshot_model_names]

        fewshot_loader_mgrs = []
        for dataset_name in data_args.fewshot_dataset_names_or_paths:
            fewshot_loader_mgrs.append(NAMES_LOADERS[dataset_name])

        fewshot_loader_mgrs = [loader_mgr(*loader_mgr_args) for loader_mgr, loader_mgr_args in zip(fewshot_loader_mgrs, fewshot_loader_mgrs_args)]
        fewshot_loaders = [loader_mgr.build_fewshot_loaders() for loader_mgr in fewshot_loader_mgrs]

        fewshot_train_loaders = [loader[0] for loader in fewshot_loaders]
        fewshot_train_subset_loaders = [loader[1] for loader in fewshot_loaders]
        fewshot_eval_loaders = [loader[2] for loader in fewshot_loaders]
        fewshot_eval_subset_loaders = [loader[3] for loader in fewshot_loaders]
        fewshot_test_loaders = [loader[4] for loader in fewshot_loaders]
        fewshot_test_subset_loaders = [loader[5] for loader in fewshot_loaders]

        return fewshot_loader_mgrs, fewshot_train_loaders, fewshot_train_subset_loaders, fewshot_eval_loaders, fewshot_eval_subset_loaders, fewshot_test_loaders, fewshot_test_subset_loaders
    
    if train_args.mode == 'train':
        logging.info("Building training and evaluation loaders")
        loader_mgrs, train_loaders, train_subset_loaders, eval_loaders, eval_subset_loaders = build_loaders(train_args)
        fewshot_loader_mgrs, fewshot_train_loaders, fewshot_train_subset_loaders, fewshot_eval_loaders, fewshot_eval_subset_loaders, fewshot_test_loaders, fewshot_test_subset_loaders = build_fewshot_loaders(train_args)

        logging.info("Building schedulers")
        total_steps = sum(len(loader) for loader in train_loaders)
        warmup_steps = train_args.warmup_steps

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
        else:
            raise ValueError("Scheduler should be either linear_warmup or cosine_warmup")

        trainer = HypernetTrainer(
            model=model, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loaders=train_loaders, 
            train_subset_loaders=train_subset_loaders, 
            eval_loaders=eval_loaders, 
            eval_subset_loaders=eval_subset_loaders,
            emb_mgrs=emb_mgrs, 
            device=device,
            loader_mgrs=loader_mgrs,
            train_args=train_args,
            fewshot_train_loaders=fewshot_train_loaders, 
            fewshot_train_subset_loaders=fewshot_train_subset_loaders, 
            fewshot_eval_loaders=fewshot_eval_loaders, 
            fewshot_eval_subset_loaders=fewshot_eval_subset_loaders, 
            fewshot_emb_mgrs=fewshot_emb_mgrs, 
            fewshot_loader_mgrs=fewshot_loader_mgrs, 
            fewshot_args=fewshot_args,
            fewshot_test_loaders=fewshot_test_loaders,
            fewshot_test_subset_loaders=fewshot_test_subset_loaders
        )

        logging.info("Starting training")
        trainer.train()
    elif train_args.mode == 'fewshot':
        fewshot_loader_mgrs, fewshot_train_loaders, fewshot_train_subset_loaders, fewshot_eval_loaders, fewshot_eval_subset_loaders, fewshot_test_loaders, fewshot_test_subset_loaders = build_fewshot_loaders(train_args)
        trainer = HypernetTrainer(
            model=model, 
            optimizer=optimizer,
            lr_scheduler=None,
            train_loaders=None, 
            train_subset_loaders=None, 
            eval_loaders=None, 
            eval_subset_loaders=None,
            emb_mgrs=None, 
            device=device,
            loader_mgrs=None,
            train_args=train_args,
            fewshot_train_loaders=fewshot_train_loaders, 
            fewshot_train_subset_loaders=fewshot_train_subset_loaders, 
            fewshot_eval_loaders=fewshot_eval_loaders, 
            fewshot_eval_subset_loaders=fewshot_eval_subset_loaders, 
            fewshot_emb_mgrs=fewshot_emb_mgrs, 
            fewshot_loader_mgrs=fewshot_loader_mgrs, 
            fewshot_args=fewshot_args,
            fewshot_test_loaders=fewshot_test_loaders,
            fewshot_test_subset_loaders=fewshot_test_subset_loaders,
        )
        logging.info("Starting fewshot training")
        trainer.fewshot_generate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    logging.info("Starting parsing")
    parser = HfArgumentParser(
        (DatasetArgs, HypnetArgs, LMArgs, MEncArgs, ProjectorArgs, TrainArgs, FewshotArgs)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, hn_args, lm_args, menc_args, projector_args, train_args, fewshot_args = parser.parse_json_file(
            json_file=osp.abspath(sys.argv[1])
        )
        name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        (
            data_args, hn_args, lm_args, menc_args, projector_args, train_args, fewshot_args
        ) = parser.parse_args_into_dataclasses()
        name = None

    args_post_init(hn_args, projector_args)

    if train_args.mode == 'train':
        init_wandb(name, 'dmi_hypernet', data_args, hn_args, lm_args, menc_args, projector_args, train_args, fewshot_args)
        main(name, train_args, hn_args, projector_args, data_args, menc_args, lm_args, fewshot_args)
        wandb.finish()
    elif train_args.mode == 'fewshot':
        seeds = train_args.seeds
        train_args.seeds = None

        epochs_l = fewshot_args.fewshot_epochs
        dataset_size_l = fewshot_args.fewshot_dataset_sizes

        for epochs, dataset_size in zip(epochs_l, dataset_size_l):
            train_args.epochs = epochs
            train_args.dataset_size = dataset_size
            logging.info(f"Training for {train_args.epochs} epochs with dataset size {train_args.dataset_size}")

            train_type = 'hypernet'
            
            for seed in seeds:
                logging.info(f"Training for seed {seed}")
                train_args.seed = seed
                set_seed(seed)

                output_fname = f"{train_type}:{name}-dsz{dataset_size}-seed{seed}"
                if osp.exists(f"../outputs/{output_fname}-results.json"):
                    logging.info(f"Skipping {output_fname} because it already exists")
                    continue

                cur_name = f"{name}-dsz{dataset_size}-seed{seed}"
                init_wandb(cur_name, 'dmi_hypernet', data_args, hn_args, lm_args, menc_args, projector_args, train_args, fewshot_args)
                main(cur_name, copy.deepcopy(train_args), copy.deepcopy(hn_args), copy.deepcopy(projector_args), copy.deepcopy(data_args), copy.deepcopy(menc_args), copy.deepcopy(lm_args), copy.deepcopy(fewshot_args))
                wandb.finish()

            if len(data_args.fewshot_dataset_names_or_paths) == 1:
                average_seed_results(seeds, name, dataset_size, data_args, train_type=train_type, field='fewshot_dataset_names_or_paths')
    else:
        raise ValueError("Mode should be either train or fewshot")