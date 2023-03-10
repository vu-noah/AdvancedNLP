# AdvancedNLP
This GitHub Repository contains files for the class Advanced NLP at VU Amsterdam. 
Authors: Noah-Manuel Michael (2778420), Basje Maes (2788012), Irma Tuinenga (2735581), Natalia Khaidanova (2778662). 

This repository contains the following directories:
- Assignment 1: directory containing code that was used for assignment 1
- Assignment 2: directory containing code that was used for assignment 2
- Data: directory holding the original .conllu files and the pre-processed files
- saved_models: directory where the fine-tuned models and the predictions will be stored

The repository contains the following files:
- requirements.txt: a file listing all the requirements to run the code
- bert_utils.py: helper functions for fine-tuning a BERT-based token classification system and making predictions
- bert_utils_reduced.py: reduced version of the above
- create_mini_files_srl_json.py: to be executed after preprocessing the connlu data sets to json, creates smaller datasets
- main.py: executes train.py and predict.py
- main_reduced.py: executes train_reduced.py and predict_reduced.py
- predict.py: contains code to make predictions on a test set with a fine-tuned model
- predict_reduced.py reduced version of the above
- preprocess_datasets.py: does not need to be executed on its own, contains code needed for preprocess_to_json.py
- preprocess_to_json.py: preprocesses the connlu data files to json format
- train.py: contains code to fine-tune a BERT-based system
- train_reduced.py: reduced version of the above

Instructions:

Call preprocess_to_json.py from the command line.

Example:

C:\\Users\\user\\Downloads\\AdvancedNLP-main\\python preprocess_to_json.py

Call create_mini_files_srl_json.py from the command line (unless you want to train and evaluate on the full dataset 
-- in this case, you need to adjust the file paths in train(_reduced).py and predict(_reduced).py to point to the 
original datasets.

Example:

C:\\Users\\user\\Downloads\\AdvancedNLP-main\\python create_mini_files_srl_json.py

Note: Our main code submissions are the scripts train.py, predict.py, and bert_utils.py.
The reduced scripts were mainly created for illustrating the blog post.

To run the main script, call the main script it in the commandline in the following format: 
[path to the current directoy] python main.py [epochs: int] [batch_size: int] [mode: str] [has_gold: bool]

There are two possible modes: token_type_IDs and flag_with_pred_token

token_type_IDs performs fine-tuning by assigning a different token type ID to the current predicate for every instance.

flag_with_pred_token performs fine-tuning by surrounding the current predicate with two special tokens: [PRED] and [\PRED].

For instance (on Windows): 

C:\\Users\\user\\Downloads\\AdvancedNLP-main\\python main.py 5 4 token_type_IDs True

Or to run the main_reduced script: 

C:\\Users\\user\\Downloads\\AdvancedNLP-main\\python main_reduced.py 1 4 token_type_IDs True

If you do want to execute the reduced scripts, within the saved_models directory, you need to manually create a new
directory: MY_BERT_SRL_reduced.
