# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from sentence_transformers import SentenceTransformer, losses, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import scipy
import math
import os

import torch
from torch.utils.data import DataLoader

from .utils import DisableLogger, get_logger

ENTAILMENT = 1.0
NON_ENTAILMENT = 0.0

logger = get_logger(__name__)

class EmbKnn:
    def __init__(self,
                 path: str,
                 args):

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        with DisableLogger():
            if path is not None and os.path.exists(path):
                self.model = SentenceTransformer(path)
            elif 'roberta' in self.args.bert_model:
                self.model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
            else:
                self.model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.model.to(self.device)
        self.cached_embeddings = None

    def save(self, dir_path):
        self.model.save(dir_path)
        
    def cache(self, example_sentences):
        self.model.eval()
        self.cached_embeddings = self.model.encode(example_sentences, show_progress_bar = False)
                
    def encode(self,
               text):
        
        self.model.eval()
        query_embeddings = self.model.encode(text, show_progress_bar = False)
        return torch.FloatTensor(query_embeddings)
        
    def predict(self,
                text):

        assert self.cached_embeddings is not None
        
        self.model.eval()

        query_embeddings = self.model.encode(text, show_progress_bar = False)
        distances = scipy.spatial.distance.cdist(query_embeddings, self.cached_embeddings, "cosine")
        distances = 1.0-distances

        return torch.FloatTensor(distances)
    
    def train(self, train_examples, dev_examples, dir_path = None):

        train_examples = SentencesDataset(train_examples, self.model)
        dev_examples = SentencesDataset(dev_examples, self.model)

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.train_batch_size)
        dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=self.args.eval_batch_size)

        train_loss = losses.CosineSimilarityLoss(model=self.model)
        evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

        warmup_steps = math.ceil(len(train_examples)*self.args.num_train_epochs/self.args.train_batch_size*self.args.warmup_proportion)

        self.model.zero_grad()
        self.model.train()
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=self.args.num_train_epochs,
                       evaluation_steps=10000,
                       warmup_steps=warmup_steps,
                       output_path=None,
                       optimizer_params = {'lr': self.args.learning_rate, 'eps': 1e-6, 'correct_bias': False})
