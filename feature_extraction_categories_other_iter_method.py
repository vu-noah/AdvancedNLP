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
    # read in the tsv file (that has no header row), assign column names, and store the data in a pandas dataframe  
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_individual_id', 'token_global_id',
                                                             'token_id_in_sent', 'token', 'lemma',
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'current_predicate', 'global_sent_id',
                                                             'candidate_prediction'],
                     quotechar='ą', engine='python')  # by setting 'quotechar' to a letter that is not part of the tsv
    # file, we make sure that nothing is counted as a quotechar (to solve the errors with punctuation chars in italics)

    # print(df)
    
    # create three empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []
    sentence_level_feature_dicts = []
            
    # check whether a token has been predicted as a candidate for an SR (if so, the value in 'candidate_prediction' = 1)
    is_candidate = df['candidate_prediction'] == 1
    candidate_df = df[is_candidate]

    for group in candidate_df.groupby('global_sent_id', sort=False):
        sent_df = group[1]
                
        # for each sentence, create two empty lists to put in the tokens (belonging to the sentence and to the
        # predicate, respectively) in later
        sentence, predicates = [], []

        count = 0  # needed to extract the argument_order feature

        # for each token in the sentence:
        for i, row in sent_df.iterrows():

            # create 2 dicts to store the categorical and numerical features in later
            categorical_feature_dict = {}
            numerical_feature_dict = {}

            # append the token and the predicate to the two empty lists
            sentence.append(row['token'].lower())
            predicates.append(row['PB_predicate'])

            # 1) extract lemma and PoS of the current token
            categorical_feature_dict['lemma'] = row['lemma'].lower()

            categorical_feature_dict['UPOS'] = row['UPOS']
            categorical_feature_dict['POS'] = row['POS']

            # 2) extract lemma and POS of the head
            head_id = row['head_id']

            if str(head_id).isdigit():
                try:
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma'].lower()
                    categorical_feature_dict['UPOS_of_head'] = head_lemmas.iloc[0]['UPOS']
                    categorical_feature_dict['POS_of_head'] = head_lemmas.iloc[0]['POS']
                except IndexError:
                    categorical_feature_dict['lemma_of_head'] = 'token_is_root'
                    categorical_feature_dict['UPOS_of_head'] = 'token_is_root'
                    categorical_feature_dict['POS_of_head'] = 'token_is_root'
            else:
                categorical_feature_dict['lemma_of_head'] = head_id.lower()
                categorical_feature_dict['UPOS_of_head'] = head_id
                categorical_feature_dict['POS_of_head'] = head_id

            # 3) obtain voice of the predicate and fill the feature dict 'voice' with the values specified below.
            # 4) obtain argument order and fill the feature dict 'predicate_order' with the values specified below.
            if row['PB_predicate'] != '_':
                count = count + 1
                if row['grammar'] == 'Tense=Past|VerbForm=Part|Voice=Pass':
                    categorical_feature_dict['voice'] = 'passive'
                    categorical_feature_dict['predicate_order'] = f'{count}_passive'

                if row['grammar'] != 'Tense=Past|VerbForm=Part|Voice=Pass':
                    categorical_feature_dict['voice'] = 'active'
                    categorical_feature_dict['predicate_order'] = f'{count}_active'
            else:  # do we need this else statement? if only candidates are being looked at
                categorical_feature_dict['voice'] = '_'
                categorical_feature_dict['predicate_order'] = '_'

            # append the feature dicts to the list
            categorical_feature_dicts.append(categorical_feature_dict)
            numerical_feature_dicts.append(numerical_feature_dict)

        # 5) get the distance from the token to the closest predicate
        # create a list of indexes that have a predicate
        predicate_indices = [i for i, predicate in enumerate(predicates) if predicate != '_']

        # for each index and token in the sentence list:
        for i, token in enumerate(sentence):

            # create an empty dict to put the distance feature in later
            distance_feature_dict = {}

            # if the token is a predicate, fill the dict with value 0
            if predicates[i] != '_':
                distance_feature_dict['distance_to_predicate'] = 0

            # otherwise, obtain the distance from the token to the closest predicate,
            # by calculating the lowest distance between the index of the token on one hand and each of the indexes of
            # the predicate_indices list on the other hand
            else:
                distance = min(abs(i - index) for index in predicate_indices)
                distance_feature_dict['distance_to_predicate'] = distance

            # append the feature dict to the list
            sentence_level_feature_dicts.append(distance_feature_dict)
      
    # return a zip with the three lists filled with feature dicts
    return zip(categorical_feature_dicts, sentence_level_feature_dicts, numerical_feature_dicts)


# extract the features to determine the SR of the candidates
if __name__ == '__main__':
    # roles_feature_dicts_train = \
    # extract_features_to_determine_roles('Data/train_data_with_candidate_predictions.tsv')
    roles_feature_dicts_test = \
        extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    # test the code
    for tup in roles_feature_dicts_test:
        print(tup)
