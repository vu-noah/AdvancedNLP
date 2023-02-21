# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def extract_features_to_determine_categories(filepath):
    """
    Extract features for SR candidate tokens (predicted in the previous step).

    :param str filepath: the path to the file with the candidate predictions
    :return: zip object (categorical_feature_dicts, numerical_feature_dicts)
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                             'upos', 'pos', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'proposition', 'semantic_role', 'is_candidate'
                                                             'candidate_prediction'])

    print(df)

    categorical_feature_dicts = []
    numerical_feature_dicts = []

    for i, token in enumerate(df['token']):
        if df['candidate_prediction'][i] == 1:

            categorical_feature_dict = {}
            numerical_feature_dict = {}

            # append the feature dicts to the list
            categorical_feature_dicts.append(categorical_feature_dict)
            numerical_feature_dicts.append(numerical_feature_dict)

    return zip(categorical_feature_dicts, numerical_feature_dicts)


if __name__ == '__main__':
    # feature_dicts_train = extract_features_to_determine_categories('Data/train_data_with_candidate_predictions.tsv')
    feature_dicts_test = extract_features_to_determine_categories('Data/test_data_with_candidate_predictions.tsv')

    # test the code
    for tup in feature_dicts_test:
        print(tup)
        break