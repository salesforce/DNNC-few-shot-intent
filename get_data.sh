# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/bin/sh

cd ./data/

# Get NLI dataset
cd ./nli/

mkdir original_data
cd ./original_data/

wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
wget https://dl.fbaipublicfiles.com/glue/data/WNLI.zip
unzip snli_1.0.zip
unzip MNLI.zip
unzip WNLI.zip

cd ..

cut -f9-10,11 ./original_data/MNLI/train.tsv > ./mnli.train.txt
cut -f6,7 ./original_data/snli_1.0/snli_1.0_train.txt > tmp1
cut -f1 ./original_data/snli_1.0/snli_1.0_train.txt > tmp2
paste tmp1 tmp2 > ./snli.train.txt
rm tmp1
rm tmp2
cut -f 2-4 ./original_data/WNLI/train.tsv > ./wnli.train.txt

cut -f9-10,11 ./original_data/MNLI/dev_matched.tsv > ./mnli.dev.txt
cut -f6,7 ./original_data/snli_1.0/snli_1.0_dev.txt > tmp1
cut -f1 ./original_data/snli_1.0/snli_1.0_dev.txt > tmp2
paste tmp1 tmp2 > ./snli.dev.txt
rm tmp1
rm tmp2
cut -f 2-4 ./original_data/WNLI/dev.tsv > ./wnli.dev.txt

python3 preprocess.py

# Get the CLINC150 dataset
cd ../clinc150/

mkdir original_data
cd ./original_data/

wget https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json

cd ..

python3 preprocess.py
