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
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_individual_id', 'token_global_id',
                                                             'token_id_in_sent', 'token', 'lemma',
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'current_predicate', 'global_sent_id'],
                     quotechar='ą', engine='python')  # by setting 'quotechar' to a letter that is not part of the tsv
    # file, we make sure that nothing is counted as a quotechar (to solve the errors with punctuation chars in italics)

    # create two empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []
    
    # create a dataframe for each sentence (i.e. rows with the same sent_id) in the same order as the original file 
    for group in df.groupby("global_sent_id", sort=False):
        sent_df = group[1]

        # get the ids of all predicates in the sentence (for checking whether the token is an immediate child)
        pb_predicate_ids = []
        for i, row in sent_df.iterrows():
            if row['PB_predicate'] != '_':
                pb_predicate_ids.append(row['token_id_in_sent'])

        # for each token in the sentence:  
        for i, row in sent_df.iterrows():
            
            # create 2 dicts to store the categorical and numerical features in later
            categorical_feature_dict = {}
            numerical_feature_dict = {}

            # 1) extract the lemma and POS of the current token
            categorical_feature_dict['lemma'] = row['lemma'].lower()

            categorical_feature_dict['UPOS'] = row['UPOS']
            categorical_feature_dict['POS'] = row['POS']
        
            # 2) extract the lemma of the head of the current token
            head_id = row['head_id']
            if str(head_id).isdigit():
                try:
                    # find row(s) in the dataframe whose token id equals the current token's head id
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma'].lower()
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

            # check if the current token is an immediate child of one of the predicates
            if row['head_id'] in pb_predicate_ids:
                numerical_feature_dict['immediate_child_of_pb_predicate'] = 1
            else:
                numerical_feature_dict['immediate_child_of_pb_predicate'] = 0

            # print(categorical_feature_dict, numerical_feature_dict)

            # append the feature dicts to the lists
            categorical_feature_dicts.append(categorical_feature_dict)
            numerical_feature_dicts.append(numerical_feature_dict)

    # return a zip with the two lists filled with feature dicts
    return df, categorical_feature_dicts, numerical_feature_dicts


# extract the features to determine the candidates
if __name__ == '__main__':
    df_train, candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train = \
        extract_features_to_determine_candidates('Data/train_data.tsv')
    df_test, candidate_cat_feature_dicts_test, candidate_num_feature_dicts_test = \
        extract_features_to_determine_candidates('Data/test_data.tsv')

    # test the code
    # for tup in candidate_feature_dicts_test:
    #     print(tup)

    run_logreg(candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train, df_train['is_candidate'].tolist(),
               candidate_cat_feature_dicts_test, candidate_num_feature_dicts_test, df_test['is_candidate'].tolist())
