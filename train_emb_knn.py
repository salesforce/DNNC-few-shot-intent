# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from intent_predictor import EmbKnnIntentPredictor

from sentence_transformers.readers.InputExample import InputExample

from models.emb_knn import EmbKnn
from models.emb_knn import ENTAILMENT, NON_ENTAILMENT

from models.utils import load_intent_datasets, load_intent_examples, sample, print_results
from models.utils import calc_oos_precision, calc_in_acc, calc_oos_recall, calc_oos_f1
from models.utils import THRESHOLDS

import argparse
from tqdm import tqdm
import json
import random
import os
from collections import defaultdict

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
    parser.add_argument("--train_batch_size",
                        default=30,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=7,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

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
    parser.add_argument('--save_model_path',
                        type = str,
                        default = '',
                        help = 'path to save the model checkpoints')

    parser.add_argument('--over_sampling',
                        type = int,
                        default = 0,
                        help = 'Over-sampling positive examples as there are more negative examples')

    parser.add_argument('--few_shot_num',
                        type = int,
                        default = 5,
                        help = 'Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type = int,
                        default = 10,
                        help = 'Number of trials to see robustness')

    parser.add_argument("--do_predict",
                        action='store_true',
                        help="do_predict the model")
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

    nli_train_examples = []
    nli_dev_examples = []

    for i in range(T):
        if args.do_predict:
            nli_train_examples.append([])
            nli_dev_examples.append([])
            continue

        tasks = sampled_tasks[i]
        all_entailment_examples = []
        all_non_entailment_examples = []

        # entailement
        for task in tasks:
            examples = task['examples']
            for j in range(len(examples)):
                for k in range(len(examples)):
                    if k <= j:
                        continue

                    all_entailment_examples.append(InputExample('', [examples[j], examples[k]], ENTAILMENT))
                    
        # non entailment
        for task_1 in range(len(tasks)):
            for task_2 in range(len(tasks)):
                if task_2 <= task_1:
                    continue
                examples_1 = tasks[task_1]['examples']
                examples_2 = tasks[task_2]['examples']
                for j in range(len(examples_1)):
                    for k in range(len(examples_2)):
                        all_non_entailment_examples.append(InputExample('', [examples_1[j], examples_2[k]], NON_ENTAILMENT))
                        
        nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
        nli_dev_examples.append(all_entailment_examples[:100] + all_non_entailment_examples[:100]) # sanity check for over-fitting
        
        for j in range(args.over_sampling):
            nli_train_examples[-1] += all_entailment_examples

    if args.output_dir is not None:
        folder_name = '{}/{}-shot-{}_nli__Based_on_nli_fined_tuned_model/'.format(args.output_dir, N, args.bert_model)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = 'batch_{}---epoch_{}---lr_{}---trials_{}'.format(args.train_batch_size,
                                                                     args.num_train_epochs,
                                                                     args.learning_rate, args.num_trials)

        file_name = '{}__oos_threshold'.format(file_name)

        if args.do_final_test:
            file_name = file_name + '_TEST.txt'
        else:
            file_name = file_name + '.txt'

        f = open(folder_name+file_name, 'w')
    else:
        f = None

    if args.save_model_path and args.do_predict:
        stats_lists_preds = defaultdict(list)

    for j in range(T):
        save_model_path = '{}_{}'.format(folder_name+args.save_model_path, j+1)
        if os.path.exists(save_model_path):
            assert args.do_predict
        else:
            assert not args.do_predict

        if args.save_model_path and os.path.exists(save_model_path):
            if args.do_predict:
                trial_stats_preds = defaultdict(list)

            model = EmbKnn(path = save_model_path,
                           args = args)
        else:
            model = EmbKnn(path = None,
                           args = args)
            model.train(nli_train_examples[j], nli_dev_examples[j])

            if args.save_model_path:
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)
                model.save(save_model_path)
            
        intent_predictor = EmbKnnIntentPredictor(model, sampled_tasks[j])
            
        in_domain_preds = []
        oos_preds = []

        for e in tqdm(dev_examples, desc = 'Intent examples'):
            pred, conf, matched_example = intent_predictor.predict_intent(e.text)
            in_domain_preds.append((conf, pred))

            if args.save_model_path and args.do_predict:
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
            pred, conf, matched_example = intent_predictor.predict_intent(e.text)
            oos_preds.append((conf, pred))

            if args.save_model_path and args.do_predict:
                if not trial_stats_preds[e.label]:
                    trial_stats_preds[e.label] = []

                single_pred = {}
                single_pred['gold_example'] = e.text
                single_pred['match_example'] = matched_example
                single_pred['gold_label'] = e.label
                single_pred['pred_label'] = pred
                single_pred['conf'] = conf
                trial_stats_preds[e.label].append(single_pred)

        if args.save_model_path and args.do_predict:
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

    if args.save_model_path and args.do_predict:
        if args.do_final_test:
            save_file = folder_name + "dev_examples_predictions_TEST.json"
        else:
            save_file = folder_name+"dev_examples_predictions.json"
        with open(save_file, "w") as outfile:
            json.dump(stats_lists_preds, outfile, indent=4)
        
if __name__ == '__main__':
    main()
