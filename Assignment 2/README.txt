# AdvancedNLP
This GitHub Repository contains files for the class Advanced NLP at VU Amsterdam.
Authors: Noah-Manuel Michael (2778420), Basje Maes (2788012), Irma Tuinenga (2735581), Natalia Khaidanova (2778662).

This repository contains the following directories:
- Assignment 1: directory containing code that was used for assignment 1
- Data: directory holding the original .conllu files
- not_used_constituency_features: a directory containing code to extract consituency features, which, due to the time
constraint, we unfortunately were not able to implement

The repository contains the following files:
- requirements.txt: a file listing all the requirements to run the code
- preprocess_datasets.py: a script used to preprocess the original .conllu files, writes the preprocessed files to
'Data' directory
- step1_feature_extraction_candidates.py: the code to extract features for a binary classifier that predicts potential
arguments of the current predicate, also trains, tests, and evaluates the classifier
- step2_feature_extraction_categories.py: the code to extract features for a multinomial classifier that predicts the
argument nature of predicted argument candidates of the current predicate, also trains, tests, and evaluates the
classifier
- main.py: the main script that executes all of the above
- Data_zipped.zip: a zip file containing the preprocessed datasets and the files with predictions
- distribution_epxeriments.py: a file containing code that would have been used to get the best distribution of classes
for each sentence, unfinished

Instructions:
- run the main script
- scores for step 1 and for step 2 will be printed to the console