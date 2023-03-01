# 21.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd
from logistic_regression_model import run_logreg


def extract_features_to_determine_candidates(filepath):
    """
    Extract features for determining whether a token is a SR candidate.
    :param str filepath: the path to the preprocessed file
    :return: tuple (df: pandas.Dataframe, categorical_feature_dicts: list[dict], numerical_feature_dicts: list[dict])
    """
    # read in the tsv file (that has no header row), assign column names, and store the data in a pandas dataframe
    df = pd.read_csv(filepath, sep='\t', header=0, quotechar='ą', engine='python')  # by setting 'quotechar' to a letter
    # that is not part of the tsv file, we make sure that nothing is counted as a quotechar (to solve the errors with
    # punctuation chars in italics)

    # create two empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []
    
    # create a dataframe for each sentence (i.e. rows with the same sent_id) in the same order as the original file 
    for group in df.groupby("global_sent_id", sort=False):
        sent_df = group[1]

        # get the ids of the predicates in the sentence (for checking whether the token is an immediate child)
        pb_predicate_ids = []
        for i, row in sent_df.iterrows():
            if row['PB_predicate'] != '_':
                pb_predicate_ids.append(row['token_id_in_sent'])

        pb_predicate_dependency_labels = []
        for i, row in sent_df.iterrows():
            if row['PB_predicate'] != '_':
                pb_predicate_dependency_labels.append(row['dependency_label'])

        index_head_dict = {token_id: head_id for token_id, head_id in
                           zip(list(sent_df['token_id_in_sent']), list(sent_df['head_id']))}

        # for each token in the sentence:  
        for i, row in sent_df.iterrows():
            
            # create 2 dicts to store the categorical and numerical features in later
            categorical_feature_dict = {}
            numerical_feature_dict = {}

            # 1) extract the lemma POS, and dependency label of the current token
            categorical_feature_dict['lemma'] = row['lemma'].lower()
            categorical_feature_dict['UPOS'] = row['UPOS']
            categorical_feature_dict['POS'] = row['POS']
            categorical_feature_dict['dependency_label'] = row['dependency_label']

            ### do here dependency_label + is_child_of_current_predicate
        
            # 2) extract the lemma of the head of the current token
            head_id = row['head_id']
            if str(head_id).isdigit():
                try:
                    # find row(s) in the dataframe whose token id equals the current token's head id
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma']
                # if the current token is the root, the above gives an IndexError; in that case we add 'None' to the
                # feature dict
                except IndexError:
                    categorical_feature_dict['lemma_of_head'] = 'token_is_root'
            else:
                categorical_feature_dict['lemma_of_head'] = head_id.lower()
            
            # extract whether the token is a NE (check whether UPOS is PROPN)
            if row['UPOS'] == 'PROPN':
                numerical_feature_dict['is_NE'] = 1
            else:
                numerical_feature_dict['is_NE'] = 0

            # check if the current token is an immediate child of the current predicate by accessing the iteration
            # number; if there are three predicates in the sentence and we look at iteration number three, then the
            # third predicate's id should be pb_predicate_ids[2]
            current_predicate_id_in_sent = pb_predicate_ids[row['iternum']]

            if row['head_id'] == current_predicate_id_in_sent:
                numerical_feature_dict['immediate_child_of_pb_predicate'] = 1
            else:
                numerical_feature_dict['immediate_child_of_pb_predicate'] = 0

            # get the dependency label of the current predicate (if it is a copula then the ARGs are usually not
            # children
            current_predicate_dependency_label = pb_predicate_dependency_labels[row['iternum']]
            categorical_feature_dict['dependency_label_of_current_predicate'] = current_predicate_dependency_label

            # check if current tokens dependency path leads up to current predicate
            token_is_nested_child = False
            # find the id of the current token
            current_token_id_in_sent = row['token_id_in_sent']
            # as long as the head_id of the current token is not the id of the current predicate,
            # we find the head of the current token
            while current_token_id_in_sent != current_predicate_id_in_sent:
                # if we reach the root (index == 0), we stop the iteration (break), means that there is no path from the
                # current token to the predicate, unless the predicate itself is the root
                if current_token_id_in_sent == 0:
                    if current_predicate_id_in_sent == 0:
                        token_is_nested_child = True
                    break
                # if we did not reach the predicate, we find the next head of the token and continue the while loop
                current_token_id_in_sent = index_head_dict[current_token_id_in_sent]
            else:  # in case we do reach the predicate, we set the value to True
                token_is_nested_child = True

            if token_is_nested_child:
                numerical_feature_dict['nested_child_of_pb_predicate'] = 1
            else:
                numerical_feature_dict['nested_child_of_pb_predicate'] = 0

            # print(categorical_feature_dict, numerical_feature_dict)

            # append the feature dicts to the lists
            categorical_feature_dicts.append(categorical_feature_dict)
            numerical_feature_dicts.append(numerical_feature_dict)

    print('Features extracted.')
    # return the df and two lists filled with feature dicts
    return df, categorical_feature_dicts, numerical_feature_dicts


# extract the features to determine the candidates
if __name__ == '__main__':
    df_train, candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train = \
        extract_features_to_determine_candidates('Data/train_data_only_current_candidates.tsv')
    df_test, candidate_cat_feature_dicts_test, candidate_num_feature_dicts_test = \
        extract_features_to_determine_candidates('Data/test_data_only_current_candidates.tsv')

    # test the code
    # for tup in candidate_feature_dicts_test:
    #     print(tup)

    run_logreg(candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train, df_train['is_candidate'].tolist(),
               candidate_cat_feature_dicts_test, candidate_num_feature_dicts_test, df_test['is_candidate'].tolist(),
               df_test, 'candidates')
