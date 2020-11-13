# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .utils import truncate_seq_pair, get_optimizer, get_train_dataloader, get_eval_dataloader, loss_with_label_smoothing, process_train_batch, accuracy
from .utils import InputExample, InputFeatures
from .utils import get_logger

import os
import random
import numpy as np
from tqdm import tqdm, trange

ENTAILMENT = 'entailment'
NON_ENTAILMENT = 'non_entailment'

logger = get_logger(__name__)

class DNNC:
    def __init__(self,
                 path: str,
                 args):

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        self.label_list = [ENTAILMENT, NON_ENTAILMENT]
        self.num_labels = len(self.label_list)

        self.config = AutoConfig.from_pretrained(self.args.bert_model, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)

        if path is not None:
            state_dict = torch.load(path+'/pytorch_model.bin')
            self.model = AutoModelForSequenceClassification.from_pretrained(path, state_dict=state_dict, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.bert_model, config=self.config)

        self.model.to(self.device)

    def save(self,
             dir_path: str):

        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model
        torch.save(model_to_save.state_dict(), '{}/pytorch_model.bin'.format(dir_path))

    def convert_examples_to_features(self, examples, train):
        label_map = {label: i for i, label in enumerate(self.label_list)}
        is_roberta = True if "roberta" in self.config.architectures[0].lower() else False

        if train:
            label_distribution = torch.FloatTensor(len(label_map)).zero_()
        else:
            label_distribution = None

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            tokens_b = self.tokenizer.tokenize(example.text_b)

            if is_roberta:
                truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 4)
            else:
                truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            if is_roberta:
                tokens_b = [self.tokenizer.sep_token] + tokens_b + [self.tokenizer.sep_token]
                segment_ids += [0] * len(tokens_b)
            else:
                tokens_b = tokens_b + [self.tokenizer.sep_token]
                segment_ids += [1] * len(tokens_b)
            tokens += tokens_b

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            if example.label is None:
                label_id = -1
            else:
                label_id = label_map[example.label]

            if train:
                label_distribution[label_id] += 1.0

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

        if train:
            label_distribution = label_distribution / label_distribution.sum()
            return features, label_distribution
        else:
            return features
        
    def train(self, train_examples, dev_examples, file_path = None):

        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        
        num_train_steps = int(
            len(train_examples) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer, scheduler = get_optimizer(self.model, num_train_steps, self.args)

        best_dev_accuracy = -1.0

        train_features, label_distribution = self.convert_examples_to_features(train_examples, train = True)
        train_dataloader = get_train_dataloader(train_features, train_batch_size)

        logger.info('***** Label distribution for label smoothing *****')
        logger.info(str(label_distribution))

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        self.model.zero_grad()
        self.model.train()
        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                input_ids, input_mask, segment_ids, label_ids = process_train_batch(batch, self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs[0]
                loss = loss_with_label_smoothing(label_ids, logits, label_distribution, self.args.label_smoothing, self.device)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()

            acc = self.evaluate(dev_examples)

            if acc > best_dev_accuracy and file_path is not None:
                best_dev_accuracy = acc
                self.save(file_path)
            
            self.model.train()

    def evaluate(self, eval_examples):
        if len(eval_examples) == 0:
            logger.info('\n  No eval data!')
            return None

        eval_features = self.convert_examples_to_features(eval_examples, train = False)
        eval_dataloader = get_eval_dataloader(eval_features, self.args.eval_batch_size)
        
        self.model.eval()
        eval_accuracy = 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)

        eval_accuracy = eval_accuracy / nb_eval_examples
        logger.info("\n  Accuracy = %f", eval_accuracy)
        return eval_accuracy

    def predict(self,
                data):

        self.model.eval()

        input = [InputExample(premise, hypothesis) for (premise, hypothesis) in data]

        eval_features = self.convert_examples_to_features(input, train = False)
        input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        max_len = input_mask.sum(dim=1).max().item()
        input_ids = input_ids[:, :max_len]
        input_mask = input_mask[:, :max_len]
        segment_ids = segment_ids[:, :max_len]

        CHUNK = 500
        EXAMPLE_NUM = input_ids.size(0)
        labels = []
        probs = None
        start_index = 0

        while start_index < EXAMPLE_NUM:
            end_index = min(start_index+CHUNK, EXAMPLE_NUM)
            
            input_ids_ = input_ids[start_index:end_index, :].to(self.device)
            input_mask_ = input_mask[start_index:end_index, :].to(self.device)
            segment_ids_ = segment_ids[start_index:end_index, :].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_, attention_mask=input_mask_, token_type_ids=segment_ids_)
                logits = outputs[0]
                probs_ = torch.softmax(logits, dim=1)

            probs_ = probs_.detach().cpu()
            if probs is None:
                probs = probs_
            else:
                probs = torch.cat((probs, probs_), dim = 0)
            labels += [self.label_list[torch.max(probs_[i], dim=0)[1].item()] for i in range(probs_.size(0))]
            start_index = end_index

        assert len(labels) == EXAMPLE_NUM
        assert probs.size(0) == EXAMPLE_NUM
            
        return labels, probs
