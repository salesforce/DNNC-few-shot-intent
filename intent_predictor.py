# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

class IntentPredictor:
    def __init__(self,
                 tasks = None):

        self.tasks = tasks

    def predict_intent(self,
                       input: str):
        raise NotImplementedError
        
class DnncIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks = None):
        
        super().__init__(tasks)
        
        self.model = model

    def predict_intent(self,
                       input: str):

        nli_input = []
        for t in self.tasks:
            for e in t['examples']:
                nli_input.append((input, e))

        assert len(nli_input) > 0

        results = self.model.predict(nli_input)
        maxScore, maxIndex = results[1][:, 0].max(dim = 0)

        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example

    
class EmbKnnIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks = None):
    
        super().__init__(tasks)
        
        self.model = model

    def predict_intent(self,
                       input: str):

        if self.model.cached_embeddings is None:
            example_sentences = []
            for t in self.tasks:
                for e in t['examples']:
                    example_sentences.append(e)
            self.model.cache(example_sentences)
        
        results = self.model.predict([input])[0]
        maxScore, maxIndex = results.max(dim = 0)
        
        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example

class TfidfKnnIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks = None):

        super().__init__(tasks)
        
        self.model = model

    def predict_intent(self,
                       input: str):

        results = self.model.predict([input])[0]
        maxScore, maxIndex = results.max(dim = 0)
        
        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example
    
class DnncJointIntentPredictor(IntentPredictor):
    def __init__(self,
                 dnnc_model,
                 knn_model,
                 tasks = None):

        super().__init__(tasks)
        
        self.dnnc_model = dnnc_model
        self.knn_model = knn_model

        self.example_num = None

        self.use_tfidf = (type(self.knn_model).__name__ == 'TfidfKnn')
        
    def build_index(self):

        self.all_candidates = []
        example_sentences = []
        for t in self.tasks:
            for e in t['examples']:
                example_sentences.append(e)
                self.all_candidates.append((e, t['task']))
        self.knn_model.cache(example_sentences)
        self.example_num = len(example_sentences)
        
    def predict_intent(self,
                       input: str,
                       topk: int):

        assert topk > 0

        topk = min(self.example_num, topk)
        
        knn_scores = self.knn_model.predict([input])
        knn_scores = knn_scores[0]
        topkScore, topkIndex = torch.topk(knn_scores, k = min(topk, knn_scores.size(0)))
        topkIndex = topkIndex.numpy()            
            
        topk_candidates = []
        nli_input = []

        for i in range(len(topkIndex)):
            topk_candidates.append(self.all_candidates[topkIndex[i]])
            nli_input.append((input, self.all_candidates[topkIndex[i]][0]))
                    
        results = self.dnnc_model.predict(nli_input)
        maxScore, maxIndex = results[1][:, 0].max(dim = 0)
        
        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        return topk_candidates[maxIndex][1], maxScore, topk_candidates[maxIndex][0]

    
