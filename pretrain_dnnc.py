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

from models.dnnc import DNNC
from models.utils import load_nli_examples

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
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.06,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--label_smoothing',
                        type = float,
                        default = 0.1,
                        help = 'Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_seq_length',
                        type = int,
                        default = 128,
                        help = 'Maximum number of paraphrases for each sentence')
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
                        required = True,
                        help = 'Validation data path')

    parser.add_argument('--model_dir_path',
                        type = str,
                        required = True,
                        help = 'Output file path')

    parser.add_argument("--do_predict",
                        action='store_true',
                        help="do_predict the model")
    
    args = parser.parse_args()

    random.seed(args.seed)

    nli_train_examples = None if args.do_predict else load_nli_examples(args.train_file_path, args.do_lower_case)
    nli_dev_examples = load_nli_examples(args.dev_file_path, args.do_lower_case)
    
    if os.path.exists('{}/pytorch_model.bin'.format(args.model_dir_path)):
        assert args.do_predict
        nli_model = DNNC(path = args.model_dir_path,
                         args = args)
        nli_model.evaluate(nli_dev_examples)

    else:
        assert not args.do_predict
        nli_model = DNNC(path = None,
                         args = args)
        if not os.path.exists(args.model_dir_path):
            os.mkdir(args.model_dir_path)
        nli_model.train(nli_train_examples, nli_dev_examples, args.model_dir_path)

        
if __name__ == '__main__':
    main()
