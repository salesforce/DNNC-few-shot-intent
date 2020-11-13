# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from models.dnnc import DNNC
from models.emb_knn import EmbKnn
from models.tfidf_knn import TfidfKnn

from intent_predictor import DnncJointIntentPredictor

from models.utils import load_intent_datasets, load_intent_examples, sample, print_results
from models.utils import calc_oos_precision, calc_in_acc, calc_oos_recall, calc_oos_f1
from models.utils import THRESHOLDS

import argparse
from tqdm import tqdm
import random
import os
from collections import defaultdict
import json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default='roberta-base',
                        type=str,
                        help="BERT model")

    parser.add_argument('--max_seq_length',
                        type = int,
                        default = 128,
                        help = 'Maximum number of paraphrases for each sentence')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    
    # Special params
    parser.add_argument('--train_file_path',
                        type = str,
                        default = None,
                        required = True,
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
                        type=str,
                        default=None,
                        required=True,
                        help='Output file path')
    parser.add_argument('--save_model_path',
                        type=str,
                        default=None,
                        help='path to save the model checkpoints')
    
    parser.add_argument('--dnnc_path',
                        type = str,
                        required = True,
                        help = 'A DNNC model checkpoint')
    parser.add_argument('--emb_knn_path',
                        type = str,
                        default = None,
                        help = 'An Emb-KNN mode checkpoint or None for TF-IDF')

    parser.add_argument("--topk",
                        default=5,
                        type=int,
                        help="Top K candidates from SBERT NLI")
    parser.add_argument('--few_shot_num',
                        type=int,
                        default=5,
                        help='Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type=int,
                        default=10,
                        help='Number of trials to see robustness')

    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_final_test the model")
    
    args = parser.parse_args()

    random.seed(args.seed)

    N = args.few_shot_num
    T = args.num_trials

    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    train_examples, dev_examples = load_intent_datasets(train_file_path, dev_file_path, args.do_lower_case)

    sampled_tasks = [sample(N, train_examples) for i in range(T)]

    if args.dev_file_path is not None:
        dev_examples = load_intent_examples(args.dev_file_path, args.do_lower_case)
    else:
        dev_examples = []
        
    if args.oos_dev_file_path is not None:
        oos_dev_examples = load_intent_examples(args.oos_dev_file_path, args.do_lower_case)
    else:
        oos_dev_examples = []

    if args.output_dir is not None:
        folder_name = '{}/{}-shot-{}_Joint---Top-{}/'.format(args.output_dir, N, args.bert_model, args.topk)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_name = '{}-shot-{}_Joint_fine_tuned_model---Top-{}'.format(N, args.bert_model, args.topk)
        if args.do_final_test:
            file_name = file_name + '_TEST.txt'
        else:
            file_name = file_name + '.txt'

        f = open(folder_name + file_name, 'w')
    else:
        f = None

    if args.save_model_path:
        stats_lists_preds = defaultdict(list)

    for j in range(T):
        trial_stats_preds = defaultdict(list)
        
        save_model_path_dnnc = '{}_{}'.format(args.dnnc_path + args.save_model_path, j + 1)
        assert os.path.exists(save_model_path_dnnc)

        if args.emb_knn_path is not None:
            save_model_path_emb_knn = '{}_{}'.format(args.emb_knn_path + args.save_model_path, j + 1)
            use_tfidf = False
            assert os.path.exists(save_model_path_emb_knn)
        else:
            use_tfidf = True

        dnnc_model = DNNC(path = save_model_path_dnnc,
                          args = args)

        if use_tfidf:
            knn_model = TfidfKnn(None)
        else:
            knn_model = EmbKnn(path = save_model_path_emb_knn,
                               args = args)

        intent_predictor = DnncJointIntentPredictor(dnnc_model,
                                                    knn_model,
                                                    sampled_tasks[j])
        intent_predictor.build_index()

        in_domain_preds = []
        oos_preds = []
        for e in tqdm(dev_examples, desc = 'Intent examples'):
            pred, conf, matched_example = intent_predictor.predict_intent(e.text,
                                                                          args.topk)
            in_domain_preds.append((conf, pred))

            if args.save_model_path:
                if not trial_stats_preds[e.label]:
                    trial_stats_preds[e.label] = []

                single_pred = {}
                single_pred['gold_example'] = e.text
                single_pred['match_example'] = matched_example
                single_pred['gold_label'] = e.label
                single_pred['pred_label'] = pred
                single_pred['conf'] = conf
                trial_stats_preds[e.label].append(single_pred)

        for e in tqdm(oos_dev_examples, desc = 'OOS examples'):
            pred, conf, matched_example = intent_predictor.predict_intent(e.text,
                                                                          args.topk)
            oos_preds.append((conf, pred))

            if args.save_model_path:
                if not trial_stats_preds[e.label]:
                    trial_stats_preds[e.label] = []

                single_pred = {}
                single_pred['gold_example'] = e.text
                single_pred['match_example'] = matched_example
                single_pred['gold_label'] = e.label
                single_pred['pred_label'] = pred
                single_pred['conf'] = conf
                trial_stats_preds[e.label].append(single_pred)

        if args.save_model_path:
            stats_lists_preds[j]=trial_stats_preds

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

    if args.save_model_path:
        if args.do_final_test:
            save_file = folder_name + "test_examples_predictions_TEST.json"
        else:
            save_file = folder_name + "dev_examples_predictions.json"

        with open(save_file, "w") as outfile:
            json.dump(stats_lists_preds, outfile, indent=4)

if __name__ == '__main__':
    main()
