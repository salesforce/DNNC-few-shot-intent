# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import glob
import sys

import torch

from models.utils import THRESHOLDS

def main():
    file_name = sys.argv[1]

    if len(sys.argv) >= 3:
        best_index = int(sys.argv[2])
    else:
        best_index = None
        
    in_acc = []
    oos_recall = []
    oos_prec = []
    oos_f1 = []
    with open(file_name, 'r') as f:
        for line in f:
            in_acc.append([])
            oos_recall.append([])
            oos_prec.append([])
            oos_f1.append([])

            for elms in line.strip().split():
                elms = elms.split(',')
                in_acc[-1].append(float(elms[0]))
                oos_recall[-1].append(float(elms[1]))
                oos_prec[-1].append(float(elms[2]))
                oos_f1[-1].append(float(elms[3]))

        in_acc = torch.FloatTensor(in_acc) * 100
        oos_recall = torch.FloatTensor(oos_recall) * 100
        oos_prec = torch.FloatTensor(oos_prec) * 100
        oos_f1 = torch.FloatTensor(oos_f1) * 100

        if best_index is None:
            best_index = (in_acc.mean(dim = 0) + oos_recall.mean(dim = 0)).max(dim = 0)[1]
        print()
        print('Best threshold: {} (index: {})'.format(THRESHOLDS[best_index], best_index))
        print('Best in_acc: {} std: {}'.format(in_acc.mean(dim = 0)[best_index], in_acc.std(dim = 0)[best_index]))
        print('Best oos_recall: {} std: {}'.format(oos_recall.mean(dim = 0)[best_index], oos_recall.std(dim = 0)[best_index]))
        print('Best oos_prec: {} std: {}'.format(oos_prec.mean(dim = 0)[best_index], oos_prec.std(dim = 0)[best_index]))
        print('Best oos_f1: {} std: {}'.format(oos_f1.mean(dim = 0)[best_index], oos_f1.std(dim = 0)[best_index]))

if __name__ == '__main__':
    main()
