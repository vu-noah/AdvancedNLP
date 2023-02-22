# 20.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def preprocess_dataset(filepath):
    """
    Preprocess a dataset in the .conllu format. Duplicate sentences with more than one event and append them at the end
    of a new file to be written. Include a new column with a binary value for whether the token is labelled as an ARG in
    the gold data. Include a column for the sentence id.
    :param str filepath: path to original dataset
    :return: None
    """
    # read in the dataset, check for longest line
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
    df = pd.read_csv(filepath, encoding='utf-8', sep='\t',
                     names=[n for n in range(longest_line_length)])

    # fill NaN values with 0s for easier processing
    df = df.fillna(0)

    # retrieve candidates
    candidates = [0 for _ in df[0]]
    for i in range(longest_line_length-11):
        target_row = 11+i
        for j, SR in enumerate(df[target_row]):
            if SR != 'V' and SR != '_' and SR != 0:
                candidates[j] = 1

    df[candidate_column_index] = candidates

    # map sent_id to tokens
    # filter dataframe for all rows that contain a sentence id and get a list of sentence ids
    is_sent_id = df[0].str.match('# sent_id = ')
    sent_ids_series = df[0][is_sent_id].tolist()

    # define a new list that will later act as a sent id column that will be appended to the df
    sent_ids_column = [sent_ids_series[0], sent_ids_series[0]]
    # iterare over the filter dataframe (starting from the first actual token)
    for boolean in is_sent_id.tolist()[2:]:
        if boolean is False:
            # if row is not a sentence id itself, take the first sentence id from the list of sentence ids and append
            # to the list that will become the new column
            sent_ids_column.append(sent_ids_series[0])
        else:  # otherwise if it is a sentence id, go to the next sent id in the sentence id and append this one
            sent_ids_series.pop(0)
            sent_ids_column.append(sent_ids_series[0])

    assert len(sent_ids_series) == 1, 'More than the final sentence id leftover.'
    assert len(df[0]) == len(sent_ids_column), 'Length of dataframe and sent_id column isn\'t the same.'

    # append new column to df
    sent_id_column = candidate_column_index+1
    df[sent_id_column] = sent_ids_column

    # retrieve the first 11 columns always, retrieve the target column (starting from 12 up to length of longest line)
    # also retrieve whether the current token is a gold candidate for a SR (i.e. labelled as an ARG)
    for i in range(longest_line_length-11):
        target_row = 11+i
        is_not_0 = df[target_row] != 0
        filtered_df = df[is_not_0]
        new_df = filtered_df.iloc[:, [n for n in range(11)] +
                                     [target_row, candidate_column_index, sent_id_column]
                                  ].copy()

        ### gets candidates only for current proposition, alternative to "retrieve candidates" block above, previous
        ## version
        # candidates = [1 if c != 'V' and c != '_' else 0 for c in new_df[target_row]]
        # new_df['is_candidate'] = candidates
        ###

        # append the filtered dataframe (containing the first 11 columns + the target column + the candidate column +
        # the sent_id column) to a csv file
        new_df.to_csv(f'Data/{datatype}_data.tsv', sep='\t', mode='a', header=False)
        print(f'{datatype.title()} dataframe with target column {target_row+1} and candidate and sent_id column '
              f'written to file.')


if __name__ == '__main__':
    preprocess_dataset('Data/en_ewt-up-train.conllu')
    preprocess_dataset('Data/en_ewt-up-test.conllu')
