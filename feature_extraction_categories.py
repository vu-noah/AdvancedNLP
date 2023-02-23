# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def extract_features_to_determine_roles(filepath):
    """
    Extract features for SR candidate tokens (predicted in the previous step).

    :param str filepath: the path to the file with the candidate predictions
    :return: zip object (categorical_feature_dicts, numerical_feature_dicts)
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                             'upos', 'pos', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'candidate_prediction'])

    #print(df)

    categorical_feature_dicts = []
    numerical_feature_dicts = []
            
    for i, token in enumerate(df['token']):
        if df['candidate_prediction'][i] == 1:
            
            for group in df.groupby('sent_id', sort = False):
                sent_df = group[1]
                
                for i, row in sent_df.iterrows():

                    categorical_feature_dict = {}
                    numerical_feature_dict = {}
                    
                    # 1) get voice of the predicate 
                    if row['grammar'] == 'Tense=Past|VerbForm=Part|Voice=Pass':
                        categorical_feature_dict['voice'] = 'passive'

                    if row['PB_predicate'] != '_' and row['grammar'] != 'Tense=Past|VerbForm=Part|Voice=Pass':
                        categorical_feature_dict['voice'] = 'active'

                    else:
                        categorical_feature_dict['voice'] = '_'

                    # append the feature dicts to the list
                    categorical_feature_dicts.append(categorical_feature_dict)
                    numerical_feature_dicts.append(numerical_feature_dict)

    return zip(categorical_feature_dicts, numerical_feature_dicts)


if __name__ == '__main__':
    # roles_feature_dicts_train = \
    # extract_features_to_determine_roles('Data/train_data_with_candidate_predictions.tsv')
    roles_feature_dicts_test = \
        extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    # test the code
    for tup in roles_feature_dicts_test:
        print(tup)
