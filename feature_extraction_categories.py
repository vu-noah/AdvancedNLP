# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd
import spacy
from spacy.tokens import Doc

nlp = spacy.load('en_core_web_sm')


def extract_features_to_determine_roles(filepath):
    """
    Extract features for SR candidate tokens (predicted in the previous step).

    :param str filepath: the path to the file with the (predicted) candidates
    :return: tuple df, categorical_feature_dicts, sentence_level_feature_dicts, numerical_feature_dicts
    """
    # read in the tsv file (that has no header row), assign column names, and store the data in a pandas dataframe  
    df = pd.read_csv(filepath, sep='\t', header=0, quotechar='ą', engine='python')  # by setting 'quotechar' to a letter
    # that is not part of the tsv file, we make sure that nothing is counted as a quotechar (to solve the errors with
    # punctuation chars in italics)

    # print(df)
    
    # create three empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []
    sentence_level_feature_dicts = []
            
    # check whether a token has been predicted as a candidate for an SR (if so, the value in 'candidate_prediction' = 1)
    if 'train' in filepath:
        candidate_column = 'is_candidate'
    elif 'test' in filepath:
        candidate_column = 'candidate_prediction'

    for i, token in enumerate(df['token']):
        if df[candidate_column][i] == 1:

            # create a dataframe for each sentence (i.e. rows with the same sent_id) in the same order as the original
            # file
            for group in df.groupby('global_sent_id', sort=False):  # does this group only have the candidate
                # predictions in them?
                sent_df = group[1]

                # Create a list of tokens and predicates, for feature 6) and 7)
                sentence = list(sent_df['token'])
                predicates = list(sent_df['PB_predicate'])
                
                count = 0 # needed to extract the argument_order feature

                # for each token in the sentence:
                for i, row in sent_df.iterrows():

                    # create 2 dicts to store the categorical and numerical features in later
                    categorical_feature_dict = {}
                    numerical_feature_dict = {}

                    # 1) extract lemma and PoS of the current token
                    categorical_feature_dict['lemma'] = row['lemma'].lower()

                    categorical_feature_dict['UPOS'] = row['UPOS']
                    categorical_feature_dict['POS'] = row['POS']

                    # 2) extract lemma and POS of the head
                    head_id = row['head_id']

                    if head_id.isdigit():
                        try:
                            # find row(s) in the dataframe whose token id equals the current token's head id
                            head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                            categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma'].lower()
                            categorical_feature_dict['UPOS_of_head'] = head_lemmas.iloc[0]['UPOS']
                            categorical_feature_dict['POS_of_head'] = head_lemmas.iloc[0]['POS']
                        # if the current token is the root, the above gives an IndexError; in that case we add 'None' to the
                        # feature dict
                        except IndexError:
                            categorical_feature_dict['lemma_of_head'] = 'token_is_root'
                            categorical_feature_dict['UPOS_of_head'] = 'token_is_root'
                            categorical_feature_dict['POS_of_head'] = 'token_is_root'
                    else:
                        categorical_feature_dict['lemma_of_head'] = head_id.lower()
                        categorical_feature_dict['UPOS_of_head'] = head_id
                        categorical_feature_dict['POS_of_head'] = head_id
                        
                    # 3) extract whether the token is a NE (check whether UPOS is PROPN)
                    if row['UPOS'] == 'PROPN':
                        numerical_feature_dict['is_NE'] = 1
                    else:
                        numerical_feature_dict['is_NE'] = 0

                    # 4) obtain voice of the predicate and fill the feature dict 'voice' with the values specified below
                    # 5) obtain predicate order and fill the feature dict 'predicate_order' with the values specified
                    # below (argument order still needs to be done)
                    if row['PB_predicate'] != '_':
                        count = count + 1
                        if row['grammar'] == 'Tense=Past|VerbForm=Part|Voice=Pass':
                            categorical_feature_dict['voice'] = 'passive'
                            categorical_feature_dict['predicate_order'] = f'{count}_passive'

                        if row['grammar'] != 'Tense=Past|VerbForm=Part|Voice=Pass':
                            categorical_feature_dict['voice'] = 'active'
                            categorical_feature_dict['predicate_order'] = f'{count}_active'
                    else:
                        categorical_feature_dict['voice'] = '_'
                        categorical_feature_dict['predicate_order'] = '_'

                    # append the feature dicts to the list
                    categorical_feature_dicts.append(categorical_feature_dict)
                    numerical_feature_dicts.append(numerical_feature_dict)

                # 6) get the distance from the token to the closest predicate
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
                    # by calculating the lowest distance between the index of the token on one hand and each of the
                    # indexes of the predicate_indices list on the other hand
                    else:
                        distance = min(abs(i - index) for index in predicate_indices)
                        distance_feature_dict['distance_to_predicate'] = distance

                    # append the feature dict to the list
                    sentence_level_feature_dicts.append(distance_feature_dict)
                
                # 7) get the NE type of the token
                # process the pretokenized text with spacy 
                doc = Doc(nlp.vocab, sentence)
                
                for token in nlp(doc):
        
                    ner_feature_dict = {}
                    # if the token is a NE, get its NE tag
                    if token.ent_type_:
                        ner_feature_dict['NE_type'] = token.ent_type_
                    # if the token is not a NE, return 'O'
                    else:
                        ner_feature_dict['NE_type'] = 'O'
                        
                    # append the feature dict to the list
                    sentence_level_feature_dicts.append(ner_feature_dict)
                    
    # return a zip with the three lists filled with feature dicts
    return df, categorical_feature_dicts, sentence_level_feature_dicts, numerical_feature_dicts


# extract the features to determine the SR of the candidates
if __name__ == '__main__':
    # roles_feature_dicts_train = \
    # extract_features_to_determine_roles('Data/train_data_only_current_candidates.tsv')
    roles_feature_dicts_test = \
        extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    # test the code
    for tup in roles_feature_dicts_test:
        print(tup)
