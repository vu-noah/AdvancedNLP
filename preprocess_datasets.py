# 20.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2

from collections import defaultdict

storage = defaultdict(list)

with open('Data/en_ewt-up-test.conllu', encoding='utf-8') as infile:
    content = infile.readlines()
    lines = [l.strip('\n').split('\t') for l in content if l != '\n']
    for line in lines:
        if not any([line[0].startswith('# newdoc id = '), line[0].startswith('# text = ')]):
            if line[0].startswith('# sent_id = '):
                current_sent_id = line
            else:
                storage[''.join(current_sent_id)].append(line)
        else:
            continue

for sent_id, tokens in storage.items():
    print(sent_id, len(tokens[0]))
