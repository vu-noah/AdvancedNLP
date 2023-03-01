# 20.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd
from collections import defaultdict


def preprocess_dataset(filepath):
    """
    Preprocess a dataset in the .conllu format. Duplicate sentences with more than one event and append them at the end
    of a new file to be written. Include a new column with a binary value for whether the token is labelled as an ARG in
    the gold data. Include a column for the sentence id. Include a column for the global sentence id. Include a column
    for the current predicate.
    :param str filepath: path to original dataset
    :return: None
    """
    # assign variables for train resp. test files, read in the dataset, check for longest line
    if 'train' in filepath:
        datatype = 'train'
    elif 'test' in filepath:
        datatype = 'test'
    else:
        print('No compatible file detected.')
        quit()

    line_lenghts = set()

    with open(filepath, encoding='utf-8') as infile:
        content = infile.readlines()
        lines = [line.strip('\n').split('\t') for line in content]
        for line in lines:
            line_lenghts.add(len(line))
    # noinspection PyUnboundLocalVariable
    print(f'Original {datatype} dataset read in.')

    longest_line_length = candidate_column_index = max(line_lenghts)
    print(f'Longest line in original dataset has {longest_line_length} columns.')

    # read in the dataset with pandas, define as many columns as there are in the longest line
    df = pd.read_csv(filepath, encoding='utf-8', sep='\t', quotechar='ą', engine='python',
                     names=[n for n in range(longest_line_length)])

    # fill NaN values with 0s for easier processing
    df = df.fillna(0)

    # create a list with zeros, the number of these zeros being equivalent to the number of rows in the dataframe, 
    # retrieve candidates for a Semantic Role (SR) for each token in the dataset (not being an empty cell, a verb, or
    # an underscore), and replace for each of these candidates the zero with an 1 in the corresponding position in the
    # list
    # candidates = [0 for _ in df[0]]
    # for i in range(longest_line_length-11):
    #     target_row = 11+i
    #     for j, SR in enumerate(df[target_row]):
    #         if SR != 'V' and SR != '_' and SR != 0:
    #             candidates[j] = 1
    # append this list to the dataframe as a new column 'candidate_column_index'. Each token that is a candidate for a
    # SR has value 1 in this column, and non-candidates have value 0.
    # df[candidate_column_index] = candidates

    # map sent_id to tokens
    # filter dataframe for all rows that contain a sentence id and get a list of sentence ids
    is_sent_id = df[0].str.match('# sent_id')
    sent_ids_series = df[0][is_sent_id].tolist()

    # define a new list that will later act as a sent id column that will be appended to the df
    sent_ids_column = [sent_ids_series[0], sent_ids_series[0]]
    
    # iterate over the filter dataframe (starting from the first actual token)
    for boolean in is_sent_id.tolist()[2:]:
        if boolean is False:
            # if row is not a sentence id itself, take the first sentence id from the list of sentence ids and append
            # to the list that will become the new column
            sent_ids_column.append(sent_ids_series[0])
        else:  # otherwise if it is a sentence id, go to the next sent id in the sentence id and append this one
            sent_ids_series.pop(0)
            sent_ids_column.append(sent_ids_series[0])
            
    # check to make sure that there is only one sentence id left in sent_ids_series
    assert len(sent_ids_series) == 1, 'More than the final sentence id leftover.'
    
    # check to make sure that the length of the dataframe and the sent_ids_column are the same
    assert len(df[0]) == len(sent_ids_column), 'Length of dataframe and sent_id column isn\'t the same.'

    # append new column 'sent_id_column' to df
    sent_id_column = candidate_column_index  # +1 for other candidate version
    df[sent_id_column] = sent_ids_column

    # retrieve the first 11 columns always, retrieve the target column (starting from 12 up to length of longest line)
    # also retrieve whether the current token is a gold candidate for an SR (i.e. labelled as an ARG)
    for i in range(longest_line_length-11):
        iternum = i
        target_row = 11+i
        is_not_0 = df[target_row] != 0
        filtered_df = df[is_not_0]
        new_df = filtered_df.iloc[:, [n for n in range(11)] +
                                     [target_row, sent_id_column]
                                  ].copy()
        # used to be: (for other candidate version)
        # new_df = filtered_df.iloc[:, [n for n in range(11)] +
        #                              [target_row, candidate_column_index, sent_id_column]
        #          ].copy()

        ## gets candidates only for current proposition, alternative to "retrieve candidates" block above, previous
        # version
        candidates = [1 if c != 'V' and c != '_' else 0 for c in new_df[target_row]]
        new_df['is_candidate'] = candidates
        new_df['iternum'] = [iternum for _ in range(len(new_df[0]))]
        ##

        # append the filtered dataframe (containing the first 11 columns + the target column + the candidate column +
        # the sent_id column) to a csv file
        new_df.to_csv(f'Data/{datatype}_data_only_current_candidates.tsv', sep='\t', mode='a', header=False)
        print(f'{datatype.title()} dataframe with target column {target_row+1} and candidate and sent_id column '
              f'written to file.')

    def add_column_for_unique_sent_id_and_current_predicate(datatype):
        """
        Take the previously written file and for each sentence, map a unique sentence id as well as the current
        predicate that is the event in this version of the sentence (if a sentence has more than one event, there are
        more than one versions of the sentence).
        :param str datatype: 'train' or 'test' for the file you are currently handling
        :return: None
        """
        if datatype == 'train':
            new_filepath = 'Data/train_data_only_current_candidates.tsv'
        elif datatype == 'test':
            new_filepath = 'Data/test_data_only_current_candidates.tsv'

        df = pd.read_csv(new_filepath, sep='\t', header=None, quotechar='ą', engine='python',
                         names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                'head_dependency_relation', 'additional_info',
                                'PB_predicate', 'semantic_role', 'sent_id', 'is_candidate', 'iternum'])  # change order
        # sent_id and is_candidate if other candidate version

        # create containers, a dictionary mapping a sentence id to all predicates that appear within this sentence, a
        # predicate column holding the current predicate that will be appended to the df, a global sent id column that
        # will map a unique sent id to every sentence, a set of sentences that have already been seen so the dictionary
        # does not hold the same predicates several times if the sentence appears more than one in the data as sentences
        # with more than one predicate have been duplicated
        predicate_sentence_mapping = defaultdict(list)
        predicate_column = []
        global_sent_id_column = []
        global_sent_id = 0
        already_seen_sent_ids = set()

        # iterate through the sent ids and map the predicates to them
        for i, sent_id in enumerate(df['sent_id']):
            try:
                if sent_id == df['sent_id'][i+1] and df['PB_predicate'][i] != '_' and sent_id not in \
                        already_seen_sent_ids:
                    predicate_sentence_mapping[sent_id].append(df['PB_predicate'][i])
                elif sent_id != df['sent_id'][i+1]:
                    if df['PB_predicate'][i] != '_' and sent_id not in already_seen_sent_ids:
                        predicate_sentence_mapping[sent_id].append(df['PB_predicate'][i])
                    already_seen_sent_ids.add(sent_id)
            except KeyError:
                print('End of dataframe.')

        # iterate through sent ids once again to fill the predicate and global sent id columns
        for i, sent_id in enumerate(df['sent_id']):
            predicate_column.append(predicate_sentence_mapping[sent_id][0])
            global_sent_id_column.append(global_sent_id)
            try:
                if sent_id != df['sent_id'][i+1]:
                    # delete a predicate from the dictionary value when the sent id changes (ie the first copy of the
                    # sentence now uses the first predicate, if the same sentence is encountered again, the second
                    # predicate will be used
                    del predicate_sentence_mapping[sent_id][0]
                    global_sent_id += 1
            except KeyError:
                print('End of dataframe.')

        # perform manual correction for a mistake in the for-loop that incorrectly labels the last sentence, no easy fix
        # found
        if datatype == 'test':
            df['current_predicate'] = predicate_column[:-65] + ['attend.01' for _ in range(65)]
            df['global_sent_id'] = global_sent_id_column[:-65] + [4798 for _ in range(65)]
        elif datatype == 'train':
            df['current_predicate'] = predicate_column[:-135] + ['hate.01' for _ in range(135)]
            df['global_sent_id'] = global_sent_id_column[:-135] + [40481 for _ in range(135)]

        # overwrite old files with new information
        df.to_csv(f'Data/{datatype}_data_only_current_candidates.tsv', sep='\t', mode='w', header=True,
                  index_label='token_individual_id')

    add_column_for_unique_sent_id_and_current_predicate(datatype)


if __name__ == '__main__':
    preprocess_dataset('../Data/en_ewt-up-train.conllu')
    preprocess_dataset('../Data/en_ewt-up-test.conllu')
