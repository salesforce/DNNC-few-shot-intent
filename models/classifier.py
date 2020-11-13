# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .utils import get_optimizer, get_train_dataloader, get_eval_dataloader, loss_with_label_smoothing, process_train_batch, accuracy
from .utils import InputExample, InputFeatures
from .utils import get_logger

import os
import random
import numpy as np
from tqdm import tqdm, trange

logger = get_logger(__name__)

class Classifier:
    def __init__(self,
                 path: str,
                 label_list,
                 args):

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        self.label_list = label_list
        self.num_labels = len(self.label_list)

        self.config = AutoConfig.from_pretrained(self.args.bert_model, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)

        if path is not None:
            state_dict = torch.load(path + '/pytorch_model.bin')
            self.model = AutoModelForSequenceClassification.from_pretrained(path, state_dict=state_dict, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.bert_model, config=self.config)

        self.model.to(self.device)

    def save(self,
             dir_path: str):

        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), '{}/pytorch_model.bin'.format(dir_path))
        
    def convert_examples_to_features(self, examples, train):
        label_map = {label: i for i, label in enumerate(self.label_list)}
        if_roberta = True if "roberta" in self.config.architectures[0].lower() else False
        
        if train:
            label_distribution = torch.FloatTensor(len(label_map)).zero_()
        else:
            label_distribution = None

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)

            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (self.args.max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            if example.label is not None:
                label_id = label_map[example.label]
            else:
                label_id = -1

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
        
    def train(self, train_examples):

        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

        num_train_steps = int(
            len(train_examples) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer, scheduler = get_optimizer(self.model, num_train_steps, self.args)

        train_features, label_distribution = self.convert_examples_to_features(train_examples, train = True)
        train_dataloader = get_train_dataloader(train_features, train_batch_size)
        
        logger.info('***** Label distribution for label smoothing *****')
        logger.info(str(label_distribution))

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self.args.train_batch_size)
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

            self.model.train()

    def evaluate(self, eval_examples):
        if len(eval_examples) == 0:
            logger.info('\n  No eval data!')
            return []

        eval_features = self.convert_examples_to_features(eval_examples, train = False)
        eval_dataloader = get_eval_dataloader(eval_features, self.args.eval_batch_size)

        self.model.eval()
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs[0]
                confs = torch.softmax(logits, dim=1)

            confs = confs.detach().cpu()

            for i in range(input_ids.size(0)):
                conf, index = confs[i].max(dim = 0)
                preds.append((conf.item(), self.label_list[index.item()]))
                
        return preds
