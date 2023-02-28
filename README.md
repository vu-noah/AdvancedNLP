# AdvancedNLP
This GitHub Repository contains files for the class Advanced NLP at VU Amsterdam. 
Authors: Noah-Manuel Michael (2778420), Basje Maes (2788012), Irma Tuinenga (2735581), Natalia Khaidanova (2778662). 

This repository contains the following directories:
- Assignment 1: directory containing code that was used for assignment 1
- Data: directory holding the original .conllu files, the preprocessed files, json files that contain features obtained from a constituency parser, as well as the files that contain the predictions made by our classifiers

The repository contains the following files:
- requirements.txt: a file listing all the requirements to run the code
- preprocess_datasets.py: a script used to preprocess the original .conllu files, writes the preprocessed files to 'Data' directory
- step1_feature_extraction_candidates.py: the code to extract features for a binary classifier that predicts potential arguments of the current predicate, also trains, tests, and evaluates the classifier
- step2_feature_extraction_categories.py: the code to extract features for a multinomial classifier that predicts the argument nature of predicted argument candidates of the current predicate, also trains, tests, and evaluates the classifier
- main.py: the main script that executes all of the above

Instructions:
- delete all the files from the 'Data' directory that are not the original .conllu files (we included them only in case there is an error with the preprocessing, which is not expected)
- run the main script
- scores for step 1 and for step 2 will be printed to the console