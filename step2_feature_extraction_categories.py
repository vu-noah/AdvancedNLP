# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd
import spacy
from spacy.tokens import Doc
from logistic_regression_model import run_logreg

nlp = spacy.load('en_core_web_lg')


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

    
    # create three empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []
            
    # check whether a token has been predicted as a candidate for an SR (if so, the value in 'candidate_prediction' = 1)
    if 'train' in filepath:
        candidate_column = 'is_candidate'
    elif 'test' in filepath:
        candidate_column = 'candidate_prediction'

    # create a dataframe for each sentence (i.e. rows with the same sent_id) in the same order as the original
    # file
    for group in df.groupby('global_sent_id', sort=False):
        sent_df = group[1]

        # A. CREATING SENTENCE LEVEL OBJECTS TO HELP WITH FEATURE EXTRACTION

        # 1. feed the sentence to a spacy pipeline, for feature 8)
        sentence = list(sent_df['token'])
        doc = Doc(nlp.vocab, sentence)
                
        # 2. create a counter for the argument candidates, for feature 5)
        argument_count = 0

        # 3. create a dictionary that maps token IDs to head IDs, for feature 9)
        index_head_dict = {token_id: head_id for token_id, head_id in zip(list(sent_df['token_id_in_sent']),
                                                                          list(sent_df['head_id']))}
        
        # 4. find information about the current predicate (each sentence is copied as many times as there are predicates
        # in the sentence;
        # each copy is linked to a specific (current) predicate for which we want to label the arguments.)
        
        # we want to know whether the current predicate is the head of the sentence, and whether or not it is passive.
        # we initially assume these to be False
        cur_pred_is_head = False
        cur_pred_is_passive = False
        
        # the iternum column holds a count that represents which copy of the sentence (and thus, which predicate) we are
        # on
        predicate_iternum = list(sent_df['iternum'])[0]
     
        # iterate over the predicate column. Our goal is to find the ith predicate for i == predicate_iternum
        counter = 0
        for i, row in sent_df.iterrows():
            # we are only interested in predicates, so we skip any non-predicates (value '_')
            if row['PB_predicate'] == '_':
                continue
            else:
                # the counter value represents one of the predicates in the sentence
                # if this counter equals the iternum, then we are on the row in which the information about the current
                # predicate is stored
                if counter == predicate_iternum:
                    # we set a variable 'cur_pred_id_in_sent' to represent the predicates id in the sentence, and
                    # 'UPOS_of_cur_pred' to represent the predicate's UPOS
                    cur_pred_id_in_sent = row['token_id_in_sent']
                    UPOS_of_cur_pred = row['UPOS']
                    # if the predicate is passive, we set cur_pred_is_passive to True
                    if "Voice=Pass" in row['grammar']:
                        cur_pred_is_passive = True
                    # if the predicate is also the head, we set cur_pred_is_head to True
                    if row['head_id'] == 0:
                        cur_pred_is_head = True
                        break
                else:
                    counter += 1

        # B. ITERATE OVER TOKENS TO ADD THE FEATURES TO THE FEATURE DICT

        # for each token in the sentence:
        for i, row in sent_df.iterrows():
            if row[candidate_column] == 1:

                # create 2 dicts to store the categorical and numerical features in later
                # we only extract features for tokens that are candidates
                categorical_feature_dict = {}
                numerical_feature_dict = {}

                # 1) extract lemma, PoS, and dependeny label of the current token
                categorical_feature_dict['lemma'] = row['lemma'].lower()

                categorical_feature_dict['UPOS'] = row['UPOS']
                categorical_feature_dict['POS'] = row['POS']
                categorical_feature_dict['dependency_label'] = row['dependency_label']

                # 2) extract lemma and POS of the head
                head_id = row['head_id']

                try:
                    # find row(s) in the dataframe whose token id equals the current token's head id
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma'].lower()
                    categorical_feature_dict['UPOS_of_head'] = head_lemmas.iloc[0]['UPOS']
                    categorical_feature_dict['POS_of_head'] = head_lemmas.iloc[0]['POS']
                    # if the current token is the root, the above gives an IndexError; in that case we add 'None' to
                    # the feature dict
                except IndexError:
                    categorical_feature_dict['lemma_of_head'] = 'token_is_root'
                    categorical_feature_dict['UPOS_of_head'] = 'token_is_root'
                    categorical_feature_dict['POS_of_head'] = 'token_is_root'

                # 3) extract whether the token is a NE (check whether UPOS is PROPN)
                if row['UPOS'] == 'PROPN':
                    numerical_feature_dict['is_NE'] = 1
                else:
                    numerical_feature_dict['is_NE'] = 0

                # 4) obtain voice of the predicate and fill the feature dict 'voice' with the values specified below
                if cur_pred_is_passive:
                    # count += 1
                    categorical_feature_dict['voice'] = 'passive'
                    # categorical_feature_dict['predicate_order'] = f'{count}_passive'
                else:
                    categorical_feature_dict['voice'] = 'active'
                    # categorical_feature_dict['predicate_order'] = '_'
                    
                #5) get argument order in combination with whether the predicate is a passive verb, active verb or other
                # each argument candidate in the sentence gets an updated value for argument_count
                argument_count += 1
                # check if the current predicate is a verb
                if UPOS_of_cur_pred in ['AUX', 'VERB']:
                    #c heck whether it is passive or active
                    if cur_pred_is_passive:
                        categorical_feature_dict['argument_order_voice'] = f"{argument_count}_passive"
                    else:
                        categorical_feature_dict['argument_order_voice'] = f"{argument_count}_active"
                # if the predicate is not a verb, assign it the category 'other'
                else:
                    categorical_feature_dict['argument_order_voice'] = f"{argument_count}_other"


                # 6) get the distance to the current predicate
                cur_index = row['token_id_in_sent']
                distance = cur_index - cur_pred_id_in_sent
                numerical_feature_dict['distance_to_predicate'] = distance

                # 7) binary feature to determine whether the token is before or after predicate
                if distance < 0:
                    # token is before the predicate
                    numerical_feature_dict['before_predicate'] = 1
                if distance >= 0:
                    # token is after the predicate
                    numerical_feature_dict['before_predicate'] = 0

                # 8) get the NE type of the token
                for token in nlp(doc):
                    # if the token is a NE, get its NE tag
                    if token.ent_type_:
                        NE_type = token.ent_type_
                # if the token is not a NE, return 'O'
                    else:
                        NE_type = 'O'

                categorical_feature_dict['NE_type'] = NE_type

                # 9) Get the dependency path from current token to current predicate
                dependency_path_to_pred = []

                # find the id of the current token
                index = row['token_id_in_sent']
                # as long as the head_id of the current token is not the id of the current predicate,
                # we find the head of the current token and add its dependency label to the list
                # 'dependency_path_to_pred'
                while index != cur_pred_id_in_sent:
                    # find the current
                    dependency_path_to_pred.append(list(sent_df['dependency_label'])[index-1])
                    # if we reach the root (index == 0), we stop the iteration (break)
                    if index == 0:
                        # if the current predicate isn't the root, that means we have not reached the current
                        # predicate, therefore there is no possible dependency path to the predicate. We create an
                        # empty path '[]'
                        if not cur_pred_is_head:
                            dependency_path_to_pred = []
                        break
                    # if we did not reach the root, we find the next head of the token and continue the while loop
                    index = index_head_dict[index]

                categorical_feature_dict['dependency_path_to_pred'] = dependency_path_to_pred
                
                # 10) add UPOS of predicate
                categorical_feature_dict['UPOS_of_cur_pred'] = UPOS_of_cur_pred

                # print(categorical_feature_dict, numerical_feature_dict)
                # append the feature dicts to the list
                categorical_feature_dicts.append(categorical_feature_dict)
                numerical_feature_dicts.append(numerical_feature_dict)

            # # 4) obtain voice of the predicate and fill the feature dict 'voice' with the values specified
            # # below
            # # 5) obtain predicate order and fill the feature dict 'predicate_order' with the values specified
            # # below (argument order still needs to be done)
            # if row['PB_predicate'] != '_':
            #     count = count + 1
            #     if row['grammar'] == 'Tense=Past|VerbForm=Part|Voice=Pass':
            #         categorical_feature_dict['voice'] = 'passive'
            #         categorical_feature_dict['predicate_order'] = f'{count}_passive'
            #
            #     if row['grammar'] != 'Tense=Past|VerbForm=Part|Voice=Pass':
            #         categorical_feature_dict['voice'] = 'active'
            #         categorical_feature_dict['predicate_order'] = f'{count}_active'
            # else:
            #     categorical_feature_dict['voice'] = '_'
            #     categorical_feature_dict['predicate_order'] = '_'

            
            
            # # 6b) get the distance from the token to the closest predicate
            
            # predicates = list(sent_df['PB_predicate'])
            
            # # create a list of indexes that have a predicate
            # predicate_indices = [i for i, predicate in enumerate(predicates) if predicate != '_']
            #
            # # for each index and token in the sentence list:
            # for i, token in enumerate(sentence):
            #
            #     # create an empty dict to put the distance feature in later
            #     distance_feature_dict = {}
            #
            #     # if the token is a predicate, fill the dict with value 0
            #     if predicates[i] != '_':
            #         distance_feature_dict['distance_to_predicate'] = 0
            #
            #     # otherwise, obtain the distance from the token to the closest predicate,
            #     # by calculating the lowest distance between the index of the token on one hand and each of the
            #     # indexes of the predicate_indices list on the other hand
            #     else:
            #         distance = min(abs(i - index) for index in predicate_indices)
            #         distance_feature_dict['distance_to_predicate'] = distance
            #
            #     # append the feature dict to the list
            #     sentence_level_feature_dicts.append(distance_feature_dict)

    print('Features extracted.')
    # print(len(categorical_feature_dicts), len(numerical_feature_dicts))
    # return the feature dicts and the dataframe
    return df, categorical_feature_dicts, numerical_feature_dicts


# extract the features to determine the SR of the candidates
if __name__ == '__main__':
    df_train, role_cat_feature_dicts_train, role_num_feature_dicts_train = \
        extract_features_to_determine_roles('Data/train_data_only_current_candidates.tsv')
    df_test, role_cat_feature_dicts_test, role_num_feature_dicts_test = \
        extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    run_logreg(role_cat_feature_dicts_train, role_num_feature_dicts_train, [role for role in
                                                                            df_train['semantic_role'] if
                                                                            role != 'V' and role != '_'],
               role_cat_feature_dicts_test, role_num_feature_dicts_test, [role for i, role in
                                                                          enumerate(df_test['semantic_role']) if
                                                                          df_test['candidate_prediction'][i] == 1],
               df_test, 'roles')  # making sure 'V' is not one of the categories we want to predict
