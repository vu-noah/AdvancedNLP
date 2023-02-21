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
    # also retrieve whether the current token is a gold candidate for a SR (i.e. labelled as an ARG)
    for i in range(longest_line_length-11):
        target_row = 11+i
        is_not_0 = df[target_row] != 0
        filtered_df = df[is_not_0]
        new_df = filtered_df.iloc[:, [n for n in range(11)] + [target_row]].copy()
        candidates = [1 if c != 'V' and c != '_' else 0 for c in new_df[target_row]]
        new_df['is_candidate'] = candidates
        # append the filtered dataframe (containing the first 11 columns + the target column + the candidate column) to
        # a csv file
        new_df.to_csv(f'Data/{datatype}_data.tsv', sep='\t', mode='a', header=False)
        print(f'{datatype.title()} dataframe with target column {target_row+1} written to file.')


if __name__ == '__main__':
    preprocess_dataset('Data/en_ewt-up-train.conllu')
    preprocess_dataset('Data/en_ewt-up-test.conllu')
    # get_gold_candidates('Data/train_data.tsv')
    # get_gold_candidates('Data/test_data.tsv')
