# 01.03.2023
# Noah-Manuel Michael
# Assignment 3

import pandas as pd
import json
import os
from preprocess_datasets import preprocess_dataset


def preprocess_dataset_to_json(filepath):
    """
    Preprocess an already preprocessed dataset in tsv format and turn it into a json file.
    :param str filepath: path to preprocessed dataset
    :return: None
    """
    # assign variables for train resp. test files, read in the dataset, check for longest line
    if 'train' in filepath:
        datatype = 'train'
    elif 'test' in filepath:
        datatype = 'test'
    else:
        raise ValueError

    df = pd.read_csv(filepath, sep='\t', header=0, quotechar='ą', engine='python')

    for group in df.groupby('global_sent_id'):
        sent_df = group[1]
        sent_dict = {'seq_words': sent_df['token'].tolist(),
                     'BIO': ['B-' + label if label != '_' else 'O' for label in sent_df['semantic_role']],
                     }

        for i, row in sent_df.iterrows():
            if row['semantic_role'] == 'V':
                sent_dict['pred_sense'] = [row['token_id_in_sent']-1, row['current_predicate']]

        if datatype == 'train':
            with open('Data/train_data.json', 'a') as outfile:
                json.dump(sent_dict, outfile)
                outfile.write('\n')
        elif datatype == 'test':
            with open('Data/test_data.json', 'a') as outfile:
                json.dump(sent_dict, outfile)
                outfile.write('\n')

    print(f'{datatype.title()} data converted to json.')


if __name__ == '__main__':
    if os.path.exists('Data/train_data_only_current_candidates.tsv'):
        preprocess_dataset_to_json('Data/train_data_only_current_candidates.tsv')
    else:
        preprocess_dataset('Data/en_ewt-up-train.conllu')
        preprocess_dataset_to_json('Data/train_data_only_current_candidates.tsv')

    if os.path.exists('Data/test_data_only_current_candidates.tsv'):
        preprocess_dataset_to_json('Data/test_data_only_current_candidates.tsv')
    else:
        preprocess_dataset('Data/en_ewt-up-test.conllu')
        preprocess_dataset_to_json('Data/test_data_only_current_candidates.tsv')

