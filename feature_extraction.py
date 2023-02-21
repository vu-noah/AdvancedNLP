# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def extract_categorical_features(filepath):
    """

    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                             'upos', 'pos', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'proposition', 'semantic_role'])

    print(df)







if __name__ == '__main__':
    extract_categorical_features('Data/test_data.tsv')
