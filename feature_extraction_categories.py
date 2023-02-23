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
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'candidate_prediction'])

    # print(df)

    categorical_feature_dicts = []
    numerical_feature_dicts = []
    sentence_level_features = []
            
    for i, token in enumerate(df['token']):
        if df['candidate_prediction'][i] == 1:
            
            for group in df.groupby('sent_id', sort = False):
                sent_df = group[1]
                
                sentence, predicates = [], []
                
                for i, row in sent_df.iterrows():

                    categorical_feature_dict = {}
                    numerical_feature_dict = {}
                    
                    sentence.append(row['token'])
                    predicates.append(row['PB_predicate'])
                    
                    # 1) extract lemma of the current token and lemma of head 
                    # 2) extract POS of the current token and POS of head 
                    categorical_feature_dict['lemma'] = row['lemma'] 

                    categorical_feature_dict['UPOS'] = row['UPOS']
                    categorical_feature_dict['POS'] = row['POS']

                    head_id = row['head_id']
                    
                    if head_id.isdigit():
                        try:
                            head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                            categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma']
                            categorical_feature_dict['UPOS_of_head'] = head_lemmas.iloc[0]['UPOS']
                            categorical_feature_dict['POS_of_head'] = head_lemmas.iloc[0]['POS']
                        except IndexError:
                            categorical_feature_dict['lemma_of_head'] = '_'
                            categorical_feature_dict['UPOS_of_head'] = '_'
                            categorical_feature_dict['POS_of_head'] = '_'
                    else:
                        categorical_feature_dict['lemma_of_head'] = head_id
                        categorical_feature_dict['UPOS_of_head'] = head_id
                        categorical_feature_dict['POS_of_head'] = head_id

                    # 3) get voice of the predicate 
                    if row['grammar'] == 'Tense=Past|VerbForm=Part|Voice=Pass':
                        categorical_feature_dict['voice'] = 'passive'

                    if row['PB_predicate'] != '_' and row['grammar'] != 'Tense=Past|VerbForm=Part|Voice=Pass':
                        categorical_feature_dict['voice'] = 'active'

                    else:
                        categorical_feature_dict['voice'] = '_'

                    # append the feature dicts to the list
                    categorical_feature_dicts.append(categorical_feature_dict)
                    numerical_feature_dicts.append(numerical_feature_dict)
                
                # 4) get the distance from the token to the closest predicate         
                predicate_indices = [i for i, predicate in enumerate(predicates) if predicate != '_']

                for i, token in enumerate(sentence):

                    distance_feature_dict = {}

                    if predicates[i] != '_':
                        distance_feature_dict['distance_to_predicate'] = 0
                    else:
                        distance = min(abs(i - index) for index in predicate_indices)
                        distance_feature_dict['distance_to_predicate'] = distance

                    # append the feature dicts to the list
                    sentence_level_features.append(distance_feature_dict)
                    
    return zip(categorical_feature_dicts, sentence_level_features, numerical_feature_dicts)


if __name__ == '__main__':
    # roles_feature_dicts_train = \
    # extract_features_to_determine_roles('Data/train_data_with_candidate_predictions.tsv')
    roles_feature_dicts_test = \
        extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    # test the code
    for tup in roles_feature_dicts_test:
        print(tup)
