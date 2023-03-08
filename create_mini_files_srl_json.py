# 06.03.2023
# Advanced NLP
# Assignment 3

# Create mini test file for the first 200 lines
with open('Data/train_data.json', 'r') as infile, open('Data/mini_train.json', 'w') as outfile:
    for i, line in enumerate(infile):
        if i < 200:
            outfile.write(line)
        else:
            break

# Create mini dev file for the following 200 lines
with open('Data/train_data.json', 'r') as infile, open('Data/mini_dev.json', 'w') as outfile:
    for i, line in enumerate(infile):
        if 200 <= i < 400:
            outfile.write(line)
        else:
            continue

# Create mini test file for the first 200 lines
with open('Data/test_data.json', 'r') as infile, open('Data/mini_test.json', 'w') as outfile:
    for i, line in enumerate(infile):
        if i < 200:
            outfile.write(line)
        else:
            break
