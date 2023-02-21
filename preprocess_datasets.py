# 20.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd


def preprocess_dataset(filepath):
    """
    Preprocess a dataset in the .conllu format. Duplicate sentences with more than one event and append them at the end
    of a new file to be written.
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
    print(f'Original {datatype} dataset read in.')

    longest_line_length = max(line_lenghts)
    print(f'Longest line in original dataset has {longest_line_length} columns.')

    # read in the dataset with pandas, define as many columns as there are in the longest line
    df = pd.read_csv(filepath, encoding='utf-8', sep='\t',
                     names=[n for n in range(longest_line_length)])

    # fill NaN values with 0s for easier processing
    df = df.fillna(0)

    # retrieve the first 11 columns always, retrieve the target column (starting from 12 up to length of longest line)
    for i in range(longest_line_length-11):
        target_row = 11+i
        is_not_0 = df[target_row] != 0
        filtered_df = df[is_not_0]
        new_df = filtered_df.iloc[:, [n for n in range(11)] + [target_row]].copy()
        # append the filtered dataframe (containing the first 11 columns + the target column and candidate column) to a
        # csv file
        new_df.to_csv(f'Data/{datatype}_data.tsv', sep='\t', mode='a', header=False)
        print(f'{datatype.title()} dataframe with target column {target_row+1} written to file.')


def get_gold_candidates(filepath):
    """
    Read in the previously processed file, check whether the token is marked as a gold candidate and store this
    information in a new column.

    :param str filepath: path to previously preprocessed file
    :return: None
    """
    if 'train' in filepath:
        datatype = 'train'
    elif 'test' in filepath:
        datatype = 'test'
    else:
        print('No compatible file detected.')
        quit()

    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'proposition', 'semantic_role'])

    candidates = []

    # check the semantic_role column, if it's not empty nor marked as 'V', token has to be a gold SR candidate
    for SR in df['semantic_role']:
        if all([SR != 'V', SR != '_']):
            candidates.append(1)
        else:
            candidates.append(0)

    df['is_candidate'] = candidates
    print('Candidates determined.')

    # overwrite the files with the dataframe that has the new column
    df.to_csv(f'Data/{datatype}_data.tsv', sep='\t', mode='w', header=False)
    print(f'{datatype.title()} dataframe with candidate column written to file.')


if __name__ == '__main__':
    preprocess_dataset('Data/en_ewt-up-train.conllu')
    preprocess_dataset('Data/en_ewt-up-test.conllu')
    get_gold_candidates('Data/train_data.tsv')
    get_gold_candidates('Data/test_data.tsv')
