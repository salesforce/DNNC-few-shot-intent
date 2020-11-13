# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch

class TfidfKnn:

    def __init__(self, example_sentences):
        self.cache(example_sentences)

    def cache(self, example_sentences):

        if example_sentences is None:
            return

        self.tfidf = TfidfVectorizer(strip_accents="unicode",
                                     stop_words="english")
        self.cached_features = self.tfidf.fit_transform(example_sentences)

    def predict(self, text):
        distances = None
        for t in text:
            text_features = self.tfidf.transform([t])
            dists = cosine_similarity(text_features, self.cached_features, "cosine").ravel()
            dists = torch.FloatTensor(dists).unsqueeze(0)
            if distances is None:
                distances = dists
            else:
                distances = torch.cat((distances, dists), dim = 0)
            
        return distances
