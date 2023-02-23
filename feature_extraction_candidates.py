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
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'current_predicate', 'global_sent_id'],
                     quotechar='ą', engine='python')


    categorical_feature_dicts = []
    numerical_feature_dicts = []
    
    for group in df.groupby("sent_id", sort = False):
        sent_df = group[1]

        # sent_df is a dataframe similar to the df above, but only contains the current sentence
        for i, row in sent_df.iterrows():

            categorical_feature_dict = {}
            numerical_feature_dict = {}

            # extract the lemma of the current token
            categorical_feature_dict['lemma'] = row['lemma']

            # extract the POS of the current token
            categorical_feature_dict['UPOS'] = row['UPOS']
            categorical_feature_dict['POS'] = row['POS']
        
            # exctract the lemma of the head of the current token
            head_id = row['head_id']
            if head_id.isdigit():
                try:
                    # find row(s) in the dataframe whose token id equals the current token's head id
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma']
                # if the current token is the root, the above gives an IndexError; in that case we add 'None' to the
                # feature dict
                except IndexError:
                    categorical_feature_dict['lemma_of_head'] = None
            else:
                categorical_feature_dict['lemma_of_head'] = head_id
            
            # extract whether the token is a NE (check whether UPOS is PROPN)
            if row['UPOS'] == 'PROPN':
                numerical_feature_dict['is_NE'] = 1
            else:
                numerical_feature_dict['is_NE'] = 0

            print(categorical_feature_dict, numerical_feature_dict)

            # append the feature dicts to the list
            categorical_feature_dicts.append(categorical_feature_dict)
            numerical_feature_dicts.append(numerical_feature_dict)

    return zip(categorical_feature_dicts, numerical_feature_dicts)


if __name__ == '__main__':
    candidate_feature_dicts_train = extract_features_to_determine_candidates('Data/train_data.tsv')
    candidate_feature_dicts_test = extract_features_to_determine_candidates('Data/test_data.tsv')

    # test the code
    for tup in candidate_feature_dicts_test:
        print(tup)
