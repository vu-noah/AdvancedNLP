# 20.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

import pandas as pd

line_lenghts = set()

with open('Data/en_ewt-up-test.conllu', encoding='utf-8') as infile:
    content = infile.readlines()
    lines = [line.strip('\n').split('\t') for line in content]
    for line in lines:
        line_lenghts.add(len(line))

longest_line_length = max(line_lenghts)

df = pd.read_csv('Data/en_ewt-up-test.conllu', encoding='utf-8', sep='\t',
                 names=[n for n in range(longest_line_length)])

df = df.fillna(0)

for i in range(longest_line_length-11):
    is_not_0 = df[10+i] != 0
    filtered_df = df[is_not_0]
    new_df = filtered_df.iloc[:, [n for n in range(11)] + [11+i]].copy()
    print(new_df)

# c11_is_not_0 = df[11] != 0
# df_1 = df[c11_is_not_0]
# df_1 = df_1.iloc[:,[n for n in range(11)] + [11]].copy()
#
# print(df_1)
#
# c12_is_not_0 = df[12] != 0
# df_2 = df[c12_is_not_0]
# df_2 = df_2.iloc[:,[n for n in range(11)] + [12]].copy()
#
# print(df_2)
#
# c13_is_not_0 = df[13] != 0
# df_3 = df[c13_is_not_0]
# df_3 = df_3.iloc[:,[n for n in range(11)] + [13]].copy()
#
# print(df_3)
