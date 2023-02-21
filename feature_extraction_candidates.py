# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def extract_features_to_determine_candidates(filepath):
    """
    Extract features for determining whether a token is a SR candidate.

    :param str filepath: the path to the preprocessed file
    :return: zip object (categorical_feature_dicts, numerical_feature_dicts)
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'proposition', 'semantic_role'])

    print(df)

    categorical_feature_dicts = []
    numerical_feature_dicts = []

    for i, token in enumerate(df['token']):

        categorical_feature_dict = {}
        numerical_feature_dict = {}

        # extract the lemma of the current token
        categorical_feature_dict['lemma'] = df['lemma'][i]

        # extract the POS of the current token
        categorical_feature_dict['UPOS'] = df['UPOS'][i]
        categorical_feature_dict['POS'] = df['POS'][i]

        # append the feature dicts to the list
        categorical_feature_dicts.append(categorical_feature_dict)
        numerical_feature_dicts.append(numerical_feature_dict)

    return zip(categorical_feature_dicts, numerical_feature_dicts)


if __name__ == '__main__':
    # feature_dicts_train = extract_features_to_determine_candidates('Data/train_data.tsv')
    feature_dicts_test = extract_features_to_determine_candidates('Data/test_data.tsv')

    # test the code
    for tup in feature_dicts_test:
        print(tup)
        break


