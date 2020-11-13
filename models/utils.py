# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import random
import numpy as np
from tabulate import tabulate
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

THRESHOLDS = [i * 0.1 for i in range(11)]

class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)

def get_logger(name):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger
        
def truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_optimizer(model, t_total, args):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    return optimizer, scheduler

def get_train_dataloader(train_features, train_batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    return train_dataloader

def get_eval_dataloader(eval_features, eval_batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    return eval_dataloader

def process_train_batch(batch, device):
    input_mask = batch[1]
    batch_max_len = input_mask.sum(dim=1).max().item()

    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    input_ids = input_ids[:, :batch_max_len]
    input_mask = input_mask[:, :batch_max_len]
    segment_ids = segment_ids[:, :batch_max_len]

    return input_ids, input_mask, segment_ids, label_ids

def loss_with_label_smoothing(label_ids, logits, label_distribution, coeff, device):
    # label smoothing
    label_ids = label_ids.cpu()
    target_distribution = torch.FloatTensor(logits.size()).zero_()
    for i in range(label_ids.size(0)):
        target_distribution[i, label_ids[i]] = 1.0
    target_distribution = coeff * label_distribution.unsqueeze(0) + (1.0 - coeff) * target_distribution
    target_distribution = target_distribution.to(device)

    # KL-div loss
    prediction = torch.log(torch.softmax(logits, dim=1))
    loss = F.kl_div(prediction, target_distribution, reduction='mean')

    return loss

class IntentExample:
    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()
        
def load_intent_examples(file_path, do_lower_case):
    examples = []

    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            e = IntentExample(text.strip(), label.strip(), do_lower_case)
            examples.append(e)

    return examples

def load_intent_datasets(train_file_path, dev_file_path, do_lower_case):
    train_examples = load_intent_examples(train_file_path, do_lower_case)
    dev_examples = load_intent_examples(dev_file_path, do_lower_case)

    return train_examples, dev_examples

def sample(N, examples):
    labels = {} # unique classes

    for e in examples:
        if e.label in labels:
            labels[e.label].append(e.text)
        else:
            labels[e.label] = [e.text]

    sampled_examples = []
    for l in labels:
        random.shuffle(labels[l])
        if l == 'oos':
            examples = labels[l][:N]
        else:
            examples = labels[l][:N]
        sampled_examples.append({'task': l, 'examples': examples})

    return sampled_examples

class InputExample(object):

    def __init__(self, text_a, text_b, label = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def load_nli_examples(file_path, do_lower_case):
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if do_lower_case:
                e = InputExample(fields[0].lower(), fields[1].lower(), fields[2])
            else:
                e = InputExample(fields[0], fields[1], fields[2])
            examples.append(e)

    return examples
        
# Evaluation metrics
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def calc_in_acc(examples, in_domain_preds, thresholds):
    in_acc = [0.0] * len(thresholds)

    for e, (conf, pred) in zip(examples, in_domain_preds):
        for i in range(len(in_acc)):
            if pred == e.label and conf >= thresholds[i]:
                in_acc[i] += 1

    if len(examples) > 0:
        for i in range(len(in_acc)):
            in_acc[i] = in_acc[i]/len(examples)

    return in_acc

def calc_oos_recall(oos_preds, thresholds):
    oos_recall = [0.0] * len(thresholds)

    for (conf, pred) in oos_preds:
        for i in range(len(oos_recall)):
            if conf < thresholds[i]:
                oos_recall[i] += 1

    if len(oos_preds) > 0:
        for i in range(len(oos_recall)):
            oos_recall[i] = oos_recall[i]/len(oos_preds)

    return oos_recall

def calc_oos_precision(in_domain_preds, oos_preds, thresholds):

    if len(oos_preds) == 0:
        return [0.0] * len(thresholds)
    
    oos_prec = []

    for th in thresholds:
        oos_output_count = 0
        oos_correct = 0

        for pred in in_domain_preds:
            if pred[0] < th:
                oos_output_count += 1

        for pred in oos_preds:
            if pred[0] < th:
                oos_output_count += 1
                oos_correct += 1

        if oos_output_count == 0:
            oos_prec.append(0.0)
        else:
            oos_prec.append(oos_correct/oos_output_count)

    return oos_prec

def calc_oos_f1(oos_recall, oos_prec):
    oos_f1 = []

    for r, p in zip(oos_recall, oos_prec):
        if r + p == 0.0:
            oos_f1.append(0.0)
        else:
            oos_f1.append(2 * r * p / (r + p))

    return oos_f1

def print_results(thresholds, in_acc, oos_recall, oos_prec, oos_f1):
    results = [['Threshold', 'In-domain accuracy', 'OOS recall', 'OOS precision', 'OOS F1']]

    for i in range(len(thresholds)):
        entry = [thresholds[i],
                 100.0 * in_acc[i],
                 100.0 * oos_recall[i],
                 100.0 * oos_prec[i],
                 100.0 * oos_f1[i]]
        results.append(entry)

    print(tabulate(results[1:], results[0], tablefmt="grid"))

