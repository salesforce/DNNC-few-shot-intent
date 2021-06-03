# DNNC-few-shot-intent

Table of contents
* <a href="https://github.com/MetaMind/DNNC-few-shot-intent#0-Introduction">0. Introduction</a>
* <a href="https://github.com/MetaMind/DNNC-few-shot-intent#1-Getting-started">1. Getting started</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#11-Install-a-pytorch-environment-by-anaconda">1.1. Install a pytorch environment by anaconda</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#12-Install-other-required-packages">1.2. Install other required packages</a></br>
--  <a href="https://github.com/MetaMind/DNNC-few-shot-intent#13-Prepare-data-for-NLI">1.3. Prepare data for NLI</a></br>
--  <a href="https://github.com/MetaMind/DNNC-few-shot-intent#14-Prepare-data-for-intent-detection">1.4. Prepare data for intent detection</a>
* <a href="https://github.com/MetaMind/DNNC-few-shot-intent#2-Models">2. Models</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#21-Train-and-evaluate-DNNC">2.1. Train and evaluate DNNC</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#22-Train-and-evaluate-Emb-kNN">2.2. Train and evaluate Emb-kNN</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#23-Train-and-evaluate-TF-IDF-kNN">2.3. Train and evaluate TF-IDF-kNN</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#24-Evaluate-DNNC-joint">2.4. Evaluate DNNC-joint</a></br>
-- <a href="https://github.com/MetaMind/DNNC-few-shot-intent#25-Train-and-evaluate-classifier">2.5. Train and evaluate classifier</a>
* <a href="https://github.com/MetaMind/DNNC-few-shot-intent#3-Questions">3. Questions?</a>
* <a href="https://github.com/MetaMind/DNNC-few-shot-intent#4-License">4. License</a>


## 0. Introduction

This is the official code base for the models in <a href="https://arxiv.org/abs/2010.13009">our paper</a> on few-shot intent detection:

* Jianguo Zhang, Kazuma Hashimoto, Wenhao Liu, Chien-Sheng Wu, Yao Wan, Philip S. Yu, Richard Socher, and Caiming Xiong. <b>Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference</b>. In EMNLP 2020. (https://arxiv.org/abs/2010.13009)

This project is a collaboration with UIC and HUST, through an internship project with <a href="https://jianguoz.github.io/">Jianguo Zhang</a>.
This code base is designed to reproduce our experiments in our paper, and can be used with other datasets.
For any technical details, please refer to <a href="https://arxiv.org/abs/2010.13009">our paper</a>.

<b>When using our code or the methods, please cite our paper: <a href="https://www.aclweb.org/anthology/2020.emnlp-main.411.bib">reference</a>.</b>

## 1. Getting started

### 1.1. Install a pytorch environment by anaconda
Other environments would work, but the following <a href="https://pytorch.org/">pytorch</a> environment is what we used in our research and development process.
```bash
conda install pytorch==1.3.0 torchvision python==3.7.2 cudatoolkit=10.0 -c pytorch
```

### 1.2. Install other required packages
```bash
pip install -r ./requirements.txt
```

### 1.3. Prepare data for NLI
To get the training data for our NLI pretraining, we can simply run
```bash
./get_data.sh
```
and then the following files are generated:
```bash
$ ls -1 ./data/nli/all*
data/nli/all_nli.dev.txt
data/nli/all_nli.train.txt
```
When we want to evaluate an NLI model on each of the SNLI, MNLI, and WNLI datasets separately, the following files would be useful:
```bash
$ ls -1 ./data/nli/*_processed.txt
data/nli/mnli.dev.txt_processed.txt
data/nli/snli.dev.txt_processed.txt
data/nli/wnli.dev.txt_processed.txt
```
Note that, there are only two classes, `entailment` and `non_entailement`, in our setup, so the NLI evaluation scores are not compatible with those of the standard three-way classification task.
Each file has the following format:
```bash
$ cat ./data/nli/all_nli.train.txt 
a person on a horse jumps over a broken down airplane.  a person is training his horse for a competition.       non_entailment
a person on a horse jumps over a broken down airplane.  a person is at a diner, ordering an omelette.   non_entailment
a person on a horse jumps over a broken down airplane.  a person is outdoors, on a horse.       entailment
children smiling and waving at camera   they are smiling at their parents       non_entailment
children smiling and waving at camera   there are children present      entailment
children smiling and waving at camera   the kids are frowning   non_entailment
...
```
, which is based on a tab-separated format: `[premise] \t [hypothesis] \t [label]`.
As long as we follow this format, we can use whatever data for the NLI pretraining step.


### 1.4. Prepare data for intent detection
If we have already done the NLI data preprocessing step, the CLIN150 dataset is ready.
If not, we can simply run
```bash
./get_data.sh
```
and then the following files are generated:
```bash
$ ls -l ./data/clinc150/ | grep '^d' | grep -v 'original_data' | cut -d" " -f 9
all
banking
credit
oos
travel
work
```
`all` corresponds to the full-domain setting with the 150 intents, `oos` corresponds to the OOS examples, and each of the others correspons to a single-domain setting.
The single-domain split is based on `./data/clinc150/domain_intent_map.json`, and this JSON file was manually created by us based on <a href="https://github.com/clinc/oos-eval/blob/master/supplementary.pdf">the CLINC150 dataset paper</a>.
Each of them has the following three directories:
```bash
$ ls -1 ./data/clinc150/banking/*
data/clinc150/banking/dev:
label
seq.in

data/clinc150/banking/test:
label
seq.in

data/clinc150/banking/train:
label
seq.in
```
and a pair of `label` and `seq.in` files look like this:
```bash
$ paste ./data/clinc150/banking/train/* | cat
transfer        i need $20000 transferred from my savings to my checking
transfer        complete a transaction from savings to checking of $20000
transfer        transfer $20000 from my savings account to checking account
transfer        take $20000 from savings and put it in checking
transfer        put $20000 into my checking account from my savings account
...
```
As long as we follow this format, we can use whatever data for the intent detection model training.

## 2. Models

Here we explain how to use the models desribed in our paper.

<b>######  Warning  ######<br>
In our code design, we sample few-shot examples every time we launch a process, so we assume that we use the same environment and the same random seed to ensure that there is consistency between the training and evaulation phases.
An alternative strategy is to create a separate set of files for pre-sampled few-shot examples, so that we can avoid performing the sampling again and again.<br>
###################</b>

### 2.1. Train and evaluate DNNC

We use the same model (`./models/dnnc.py`) to pretrain our DNNC model with NLI and then train it for the intent detection task.

* <b>Pretraining with NLI</b>

The following command shows how to pretrain our DNNC model with the NLI dataset created by `./get_data.sh` as described above:
```bash
python pretrain_dnnc.py \
--train_file_path ./data/nli/all_nli.train.txt \
--dev_file_path ./data/nli/all_nli.dev.txt \
--do_lower_case \
--model_dir_path ./roberta_nli/
```
, where we set the same hyper-parameter setting used in our paper.
In this example, a RoBERTa-based NLI model will be saved as `./roberta_nli/pytorch_model.bin`.
If we want to evaluate the model again, we can simply add the `--do_predict` option.
Alternatively, we also provide our own RoBERTa NLI model at <a href="https://storage.googleapis.com/sfr-dnnc-few-shot-intent/roberta_nli.zip">this URL</a> (`roberta_nli.zip`) to skip this pretraining step.

* <b>Training for intent detection</b>

The following command shows how to train our DNNC model with the CLINC150 dataset created by `./get_data.sh` as described above:
```bash
python train_dnnc.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/dev/ \
--oos_dev_file_path ./data/clinc150/oos/dev/ \
--do_lower_case \
--bert_nli_path ./roberta_nli/ \
--bert_model roberta-base \
--few_shot_num 5 \
--num_trials 5 \
--num_train_epochs 10 \
--learning_rate 2e-5 \
--train_batch_size 400 \
--gradient_accumulation_steps 4 \
--save_model_path saving_checkpoints \
--output_dir ./clinc150_banking_dnnc/
```
, where we use the banking domain as an example, assuming that we have a NLI-pretrained model at `./roberta_nli/`.
If we do not want to use any NLI pretrained models, we can replace `--bert_nli_path ./roberta_nli/` with `--scratch`, so that we can train an intent detetion model directly from the original BERT/RoBERTa models.
`--few_shot_num 5` and `--num_trials 5` together specify a setting of 5 trials of 5-shot learning; the 5 examples of a class in each trial is randomly sampled from the training dataset.
If we want to use a fixed example set, we can create such a dataset with `K` examples for each class and set `--few_shot_num K` and `--num_trials 1`.
After each trial of the training, we can see a result table like this:
```bash
+-------------+----------------------+--------------+-----------------+----------+
|   Threshold |   In-domain accuracy |   OOS recall |   OOS precision |   OOS F1 |
+=============+======================+==============+=================+==========+
|         0   |              95.3333 |            0 |          0      |   0      |
+-------------+----------------------+--------------+-----------------+----------+
|         0.1 |              93.6667 |           88 |         90.7216 |  89.3401 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.2 |              93      |           90 |         89.1089 |  89.5522 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.3 |              93      |           90 |         88.2353 |  89.1089 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.4 |              93      |           91 |         88.3495 |  89.6552 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.5 |              92.6667 |           94 |         87.8505 |  90.8213 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.6 |              92      |           97 |         86.6071 |  91.5094 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.7 |              91.6667 |           97 |         85.8407 |  91.0798 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.8 |              90      |           98 |         82.3529 |  89.4977 |
+-------------+----------------------+--------------+-----------------+----------+
|         0.9 |              86.6667 |           99 |         74.4361 |  84.9785 |
+-------------+----------------------+--------------+-----------------+----------+
|         1   |               0      |          100 |         25      |  40      |
+-------------+----------------------+--------------+-----------------+----------+
```
, where all the OOS-related scores are 0 if we do not use any OOS evaluation sets.
Once all the training processes are completed, we can see the following file and directories:
```bash
$ ls -1 ./clinc150_banking_dnnc/5-shot-roberta-base_nli__Based_on_nli_fine_tuned_model/
batch_400---epoch_10.0---lr_2e-05---trials_5__oos-threshold__based_on_nli_fine_tuned_model.txt
saving_checkpoints_1
saving_checkpoints_2
saving_checkpoints_3
saving_checkpoints_4
saving_checkpoints_5
```
, where the first text file stores all the evaluation scores on the dev set, and if we set `--save_model_path saving_checkpoints`, all the trained models are saved in the `saving_checkpoints_X` directories.
To calculate the statistics of the socres as reported in our paper, we can run the following command:
```bash
$ python calc_stats.py \
./clinc150_banking_dnnc/5-shot-roberta-base_nli__Based_on_nli_fine_tuned_model/batch_400---epoch_10.0---lr_2e-05---trials_5__oos-threshold__based_on_nli_fine_tuned_model.txt
```
, and we can see results like these:
```bash
Best threshold: 0.7000000000000001 (index: 7)
Best in_acc: 91.86666107177734 std: 2.8244941234588623
Best oos_recall: 97.5999984741211 std: 0.8944271802902222
Best oos_prec: 86.95243072509766 std: 3.594482183456421
Best oos_f1: 91.93324279785156 std: 1.8878377676010132
```

Finally, we can evalaute the models on the test set, by running the following command with `--do_predict`:
```bash
python train_dnnc.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/test/ \
--oos_dev_file_path ./data/clinc150/oos/test/ \
--do_lower_case \
--bert_nli_path ./roberta_nli/ \
--bert_model roberta-base \
--few_shot_num 5 \
--num_trials 5 \
--num_train_epochs 10 \
--save_model_path saving_checkpoints \
--output_dir ./clinc150_banking_dnnc/ \
--do_predict \
--do_final_test
```
, and the following file will be generated:
```bash
./clinc150_banking_dnnc/5-shot-roberta-base_nli__Based_on_nli_fined_tuned_model/batch_400---epoch_10.0---lr_2e-05---trials_5__oos-threshold__based_on_nli_fine_tuned_model_TEST.txt
```
This file stores the evaluation scores on the test set, and we can apply the same script to get the statistics by specifying the best threshold value identified above:
```bash
$ python calc_stats.py \
./clinc150_banking_dnnc/5-shot-roberta-base_nli__Based_on_nli_fine_tuned_model/batch_400---epoch_10.0---lr_2e-05---trials_5__oos-threshold__based_on_nli_fine_tuned_model_TEST.txt \
7
```


### 2.2. Train and evaluate Emb-kNN

To train the Emb-kNN model, we do not need to pretrain it with the NLI dataset, becasuse we directly use NLI-pretrained sentence transformer models from the <a href="https://github.com/UKPLab/sentence-transformers">sentence-transformers</a> library.
The following command shows how to train the model with the same dataset as in the DNNC example:
```bash
python train_emb_knn.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/dev/ \
--oos_dev_file_path ./data/clinc150/oos/dev/ \
--do_lower_case \
--bert_model roberta-base \
--few_shot_num 5 \
--num_trials 5 \
--num_train_epochs 25 \
--learning_rate 2e-5 \
--train_batch_size 100 \
--save_model_path saving_checkpoints \
--output_dir ./clinc150_banking_embknn/
```
, where semantics of the arguments is consistent with the DNNC scenario's.
The remaining evaluation process is exactly the same as before.

### 2.3. Train and evaluate TF-IDF-kNN

We also provide the TF-IDF retrieval baseline, and the following command shows an example of how to run it:
```bash
python train_tfidf_knn.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/dev/ \
--oos_dev_file_path ./data/clinc150/oos/dev/ \
--do_lower_case \
--few_shot_num 5 \
--num_trials 5 \
--output_dir ./clinc150_banking_tfidf/
```
, where the training process is much simpler than before, because this just builds a TF-IDF index.
The remaining evaluation process is almost the same as before, except that we do not save any models and we can just add `--do_final_test` to get scores for the test set.

### 2.4. Evaluate DNNC-joint

One we train the DNNC model and the Emb-kNN model, we can use our DNNC-joint method.
An example command looks like this:
```bash
python run_dnnc_joint.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/dev/ \
--oos_dev_file_path ./data/clinc150/oos/dev/ \
--do_lower_case \
--bert_model roberta-base \
--few_shot_num 5 \
--num_trials 5 \
--save_model_path saving_checkpoints \
--output_dir ./clinc150_banking_joint/ \
--dnnc_path ./clinc150_banking_dnnc/5-shot-roberta-base_nli__Based_on_nli_fine_tuned_model/ \
--emb_knn_path ./clinc150_banking_embknn/5-shot-roberta-base_nli__Based_on_nli_fined_tuned_model/ \
--topk 20
```
, where we assume that both the DNNC and Emb-kNN models are trained in an identical setting.
In our paper, we only used the Emb-kNN model for the fast retrieval compoenent, but here we also make it possible to use the TF-IDF retrieval as well.
We can simply remove the `--emb_knn_path` argument to enable the TF-IDF option.
The remaining evaluation process is almost the same as before, except that we can just add `--do_final_test` to get scores for the test set.


### 2.5. Train and evaluate classifier

As the most standard baseline, the following command shows how to use the softmax classifier baseline:
```bash
python train_classifier.py \
--train_file_path ./data/clinc150/banking/train/ \
--dev_file_path ./data/clinc150/banking/dev/ \
--oos_dev_file_path ./data/clinc150/oos/dev/ \
--do_lower_case \
--bert_model roberta-base \
--few_shot_num 5 \
--num_trials 5 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--train_batch_size 30 \
--save_model_path saving_checkpoints \
--output_dir ./clinc150_banking_classifier/
```
, where semantics of the arguments is consistent with the others.
The remaining evaluation process is exactly the same as before.

## More few-shot learning results on other datasets

We also report average few-shot learning results without oos intent detection across five different runs on [CLINC150](https://www.aclweb.org/anthology/D19-1131.pdf), [BANKING77](https://www.aclweb.org/anthology/2020.nlp4convai-1.5.pdf) and [HWU64](https://arxiv.org/pdf/1903.05566.pdf). 

```bash
+-------------+----------------------+--------------+
|   Datasets|   CLINC150 |   HWU64 |   BANKING77 |   
+=============+======================+==============+=================+
|        USE+CONVRT   |              90.49 |      80.01       |     77.75|         
+-------------+----------------------+--------------+-----------------+
|         DNNC |         91.02       |    80.46        |     80.40     |  
+-------------+----------------------+--------------+-----------------+

```


## 4. Questions?
For any questions, feel free to open issues, or shoot emails to

* Kazuma Hashimoto (k.hashimoto@salesforce.com)
*  <a href="https://jianguoz.github.io/">Jianguo Zhang</a>

## 5. License

* <a href="./LICENSE.txt">MIT</a>
