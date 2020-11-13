# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
from tqdm import tqdm
import random
import os

from models.tfidf_knn import TfidfKnn

from models.utils import load_intent_datasets, load_intent_examples, sample, print_results
from models.utils import calc_oos_precision, calc_in_acc, calc_oos_recall, calc_oos_f1
from models.utils import THRESHOLDS

from intent_predictor import TfidfKnnIntentPredictor

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")

    # Special params
    parser.add_argument('--train_file_path',
                        type = str,
                        default = None,
                        help = 'Training data path')
    parser.add_argument('--dev_file_path',
                        type = str,
                        default = None,
                        help = 'Validation data path')
    parser.add_argument('--oos_dev_file_path',
                        type = str,
                        default = None,
                        help = 'Out-of-Scope validation data path')

    parser.add_argument('--output_dir',
                        type = str,
                        default = None,
                        help = 'Output file path')
    
    parser.add_argument('--few_shot_num',
                        type = int,
                        default = 5,
                        help = 'Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type = int,
                        default = 10,
                        help = 'Number of trials to see robustness')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_predict the model")

    args = parser.parse_args()

    random.seed(args.seed)

    N = args.few_shot_num
    T = args.num_trials

    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    train_examples, dev_examples = load_intent_datasets(train_file_path, dev_file_path, args.do_lower_case)

    sampled_tasks = [sample(N, train_examples) for i in range(T)]

    if args.oos_dev_file_path is not None:
        oos_dev_examples = load_intent_examples(args.oos_dev_file_path, args.do_lower_case)
    else:
        oos_dev_examples = []

    if args.output_dir is not None:
        folder_name = '{}/{}-shot-TF-IDF/'.format(args.output_dir, N)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = 'trials_{}'.format(args.num_trials)
        file_name = '{}__oos_threshold'.format(file_name)

        if args.do_final_test:
            file_name = file_name + '_TEST.txt'
        else:
            file_name = file_name + '.txt'

        f = open(folder_name+file_name, 'w')
    else:
        f = None
        
    for j in range(T):
        example_sentences = []
        for t in sampled_tasks[j]:
            for e in t['examples']:
                example_sentences.append(e)
        model = TfidfKnn(example_sentences)
        intent_predictor = TfidfKnnIntentPredictor(model,
                                                   sampled_tasks[j])
            
        in_domain_preds = []
        oos_preds = []

        for e in dev_examples:
            pred, conf, matched_example = intent_predictor.predict_intent(e.text)
            in_domain_preds.append((conf, pred))

        for e in oos_dev_examples:
            pred, conf, matched_example = intent_predictor.predict_intent(e.text)
            oos_preds.append((conf, pred))

        in_acc = calc_in_acc(dev_examples, in_domain_preds, THRESHOLDS)
        oos_recall = calc_oos_recall(oos_preds, THRESHOLDS)
        oos_prec = calc_oos_precision(in_domain_preds, oos_preds, THRESHOLDS)
        oos_f1 = calc_oos_f1(oos_recall, oos_prec)

        print_results(THRESHOLDS, in_acc, oos_recall, oos_prec, oos_f1)

        if f is not None:
            for i in range(len(in_acc)):
                f.write('{},{},{},{} '.format(in_acc[i], oos_recall[i], oos_prec[i], oos_f1[i]))
            f.write('\n')

    if f is not None:
        f.close()

        
if __name__ == '__main__':
    main()
