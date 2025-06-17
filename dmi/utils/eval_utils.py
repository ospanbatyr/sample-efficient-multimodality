import os
import json
import string
import pickle
import evaluate
import numpy as np
from tqdm import tqdm
import os.path as osp
from rouge_score import rouge_scorer
from cococap.pycocotools.coco import COCO
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from cococap.pycocoevalcap.eval import COCOEvalCap
from nltk.translate.meteor_score import meteor_score
def calculate_bleu(preds, gts, experiment_id):
    bleu = evaluate.load("bleu", experiment_id=experiment_id)
    all_bleus = []
    for i in range(4):
        bleu_score = bleu.compute(predictions=preds, references=gts, max_order=i+1)["bleu"]
        all_bleus.append(bleu_score)

    return all_bleus

def caption_evaluate_chebi20(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    
    for i, (gt, out) in enumerate(tqdm(zip(targets, predictions))):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu *= 100

    print('BLEU score:', bleu)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return {'rouge1': rouge_1, 'rouge2': rouge_2, 'rougeL': rouge_l, 'bleu': bleu, 'meteor': _meteor_score}


def caption_evaluate(preds, gts, experiment_id, dataset_name):
    if dataset_name == 'chebi20':
        eval_tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')
    else:
        eval_tokenizer = None

    print(f'Using {eval_tokenizer} for evaluation')

    rouge_eval = evaluate.load('rouge', experiment_id=experiment_id)
    bleu_eval = evaluate.load('bleu', experiment_id=experiment_id)
    meteor_eval = evaluate.load('meteor', experiment_id=experiment_id)

    if eval_tokenizer is not None:
        rouge_scores = rouge_eval.compute(predictions=preds, references=gts, tokenizer=eval_tokenizer)
        bleu_scores = bleu_eval.compute(predictions=preds, references=gts, tokenizer=eval_tokenizer)
        meteor_scores = meteor_eval.compute(predictions=preds, references=gts, tokenizer=eval_tokenizer)
    else:
        rouge_scores = rouge_eval.compute(predictions=preds, references=gts)
        bleu_scores = bleu_eval.compute(predictions=preds, references=gts)
        meteor_scores = meteor_eval.compute(predictions=preds, references=gts)
    return {**rouge_scores, 'bleu': bleu_scores['bleu'], **meteor_scores}


def load_chebi_gts(split):
    chebi_gts = dict()
    
    with open(f'data/chebi20/chebi_{split}.txt', 'r') as f:
        lines = [line.strip().strip(string.punctuation) for line in f][1:]
        for line in lines:
            cid, _, desc = line.split('\t')
            chebi_gts[cid] = desc

    return chebi_gts

def load_sydney_gts(split):
    sydney_gts = dict()
    with open('data/sydney/dataset_sydney.json', 'r') as f:
        items = json.load(f)['images']
        for item in items:
            if item['split'] == split:
                cid = str(item['imgid'])
                captions = [i['raw'].strip(' .') for i in item['sentences']]
                sydney_gts[cid] = captions

    return sydney_gts

def load_candels_gts(split):
    candels_gts = dict()
    with open(f'data/candels/{split}_embs_gte-modernbert-base.pkl', 'rb') as f:
        text_embs = pickle.load(f)
    
    for full_id, caption in text_embs.keys():
        imgid = f'{full_id.split("_")[0]}_{full_id.split("_")[1]}'
        if imgid in candels_gts:
            candels_gts[imgid].append(caption)
        else:
            candels_gts[imgid] = [caption]

    return candels_gts

def calc_metrics(preds, ids, dataset_name, experiment_id, mode):
    img_ids = []

    for image_id in ids:
        image_id_parts = image_id.split('_')
        if len(image_id_parts) in [1, 2]:
            image_id = image_id_parts[0]
        elif len(image_id_parts) == 3:
            image_id = f"{image_id_parts[0]}_{image_id_parts[1]}"
        else:
            raise ValueError(f"Invalid image_id:'{image_id}'")

        img_ids.append(image_id)

    if dataset_name == 'chebi20':
        split = dict(eval='validation', test='test')[mode]
        gts = load_chebi_gts(split)
    elif dataset_name == 'sydney':
        split = dict(eval='val', test='test')[mode]
        gts = load_sydney_gts(split)
    elif dataset_name == 'candels':
        split = dict(eval='validation', test='test')[mode]
        gts = load_candels_gts(split)
    
    new_preds = []
    new_gts = []
    for pred, img_id in zip(preds, img_ids):
        new_preds.append(pred)
        new_gts.append(gts[img_id])

    if dataset_name == 'chebi20':
        eval_tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')
        text_trunc_length = 802 # 802 is the max length * 2 of the chebi20 dataset
        metrics = caption_evaluate_chebi20(new_preds, new_gts, eval_tokenizer, text_trunc_length)
    else:
        metrics = caption_evaluate(new_preds, new_gts, experiment_id, dataset_name)

    if dataset_name in ['candels', 'sydney']:
        coco_cider, coco_bleu, coco_meteor, coco_rouge = calc_cider(preds, img_ids, dataset_name, split, experiment_id)
        metrics['coco_cider'] = coco_cider
        metrics['coco_bleu'] = coco_bleu
        metrics['coco_meteor'] = coco_meteor
        metrics['coco_rouge'] = coco_rouge

    return metrics
    
def calc_cider(preds, img_ids, dataset_name, split, experiment_id):
    data = []
    for pred, img_id in zip(preds, img_ids):
        data.append({
            'image_id': img_id,
            'caption': pred
        })

    # create a temporary json file and save the data
    with open(f'temp_{experiment_id}.json', 'w') as f:
        json.dump(data, f)
    
    coco = COCO(f'data/{dataset_name}/{dataset_name}_{split}_annotations.json')
    cocoRes = coco.loadRes(f'temp_{experiment_id}.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    coco_cider = cocoEval.eval['CIDEr']
    coco_bleu = cocoEval.eval['Bleu_4']
    coco_meteor = cocoEval.eval['METEOR']
    coco_rouge = cocoEval.eval['ROUGE_L']
        
    if osp.exists(f'temp_{experiment_id}.json'):
        os.remove(f'temp_{experiment_id}.json')

    return coco_cider, coco_bleu, coco_meteor, coco_rouge
