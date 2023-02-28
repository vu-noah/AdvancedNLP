# 28.02.2023
# Noah-Manuel Michael
# Advanced NLP Assignment 2
# This script does not work yet, unfortunately we had to give up due to time constraints

import pandas as pd
from itertools import permutations

df = pd.read_csv('Data/test_data_with_role_predictions.tsv', sep='\t', header=0, quotechar='ą', engine='python')


### this function was generated with the help of ChatGPT
def find_best_combination(prob_lists, class_labels):
    """
    Given a list of lists of probabilities representing a class, find the best combination of classes.
    The number of classes is equal to the number of probability lists.
    Each class may appear only once in the best combination.
    The function returns a tuple of class labels that correspond to the best combination, as well as the probability of that combination.
    """
    num_classes = len(prob_lists)
    class_indices = range(len(class_labels))
    best_prob = 0.0
    best_combination = None

    for combination in permutations(class_indices, num_classes):
        prob = 1.0
        used_classes = set()
        for i, p in enumerate(prob_lists):
            class_index = combination[i]
            class_label = class_labels[class_index]
            if class_label in used_classes:
                prob = 0.0
                break
            prob *= p[class_index]
            used_classes.add(class_label)
        if prob > best_prob:
            best_prob = prob
            best_combination = combination

    return tuple(class_labels[i] for i in best_combination), best_prob
###


class_names = ['ARG0', 'ARG1', 'ARG1-DSP', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU',
               'ARGM-COM', 'ARGM-CXN', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB',
               'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-REC', 'ARGM-TMP',
               'C-ARG0', 'C-ARG1', 'C-ARG1-DSP', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-CXN',
               'C-ARGM-DIR', 'C-ARGM-EXT', 'C-ARGM-GOL', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-PRP', 'C-ARGM-PRR',
               'C-ARGM-TMP', 'C-V', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU',
               'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-TMP']

for group in df.groupby('global_sent_id'):
    sent_df = group[1]

    list_ARG0 = []
    list_ARG1 = []
    list_ARG1_DSP = []
    list_ARG2 = []
    list_ARG3 = []
    list_ARG4 = []
    list_ARG5 = []
    list_ARGA = []
    list_ARGM_ADJ = []
    list_ARGM_ADV = []
    list_ARGM_CAU = []
    list_ARGM_COM = []
    list_ARGM_CXN = []
    list_ARGM_DIR = []
    list_ARGM_DIS = []
    list_ARGM_EXT = []
    list_ARGM_GOL = []
    list_ARGM_LOC = []
    list_ARGM_LVB = []
    list_ARGM_MNR = []
    list_ARGM_MOD = []
    list_ARGM_NEG = []
    list_ARGM_PRD = []
    list_ARGM_PRP = []
    list_ARGM_PRR = []
    list_ARGM_REC = []
    list_ARGM_TMP = []
    list_C_ARG0 = []
    list_C_ARG1 = []
    list_C_ARG1_DSP = []
    list_C_ARG2 = []
    list_C_ARG3 = []
    list_C_ARG4 = []
    list_C_ARGM_ADV = []
    list_C_ARGM_COM = []
    list_C_ARGM_CXN = []
    list_C_ARGM_DIR = []
    list_C_ARGM_EXT = []
    list_C_ARGM_GOL = []
    list_C_ARGM_LOC = []
    list_C_ARGM_MNR = []
    list_C_ARGM_PRP = []
    list_C_ARGM_PRR = []
    list_C_ARGM_TMP = []
    list_C_V = []
    list_R_ARG0 = []
    list_R_ARG1 = []
    list_R_ARG2 = []
    list_R_ARG3 = []
    list_R_ARG4 = []
    list_R_ARGM_ADV = []
    list_R_ARGM_CAU = []
    list_R_ARGM_COM = []
    list_R_ARGM_DIR = []
    list_R_ARGM_GOL = []
    list_R_ARGM_LOC = []
    list_R_ARGM_MNR = []
    list_R_ARGM_TMP = []

    for i, row in sent_df.iterrows():

        if row['candidate_prediction'] == 1:
            list_ARG0.append(float(row["ARG0"]))
            list_ARG1.append(float(row["ARG1"]))
            list_ARG1_DSP.append(float(row["ARG1-DSP"]))
            list_ARG2.append(float(row["ARG2"]))
            list_ARG3.append(float(row["ARG3"]))
            list_ARG4.append(float(row["ARG4"]))
            list_ARG5.append(float(row["ARG5"]))
            list_ARGA.append(float(row["ARGA"]))
            list_ARGM_ADJ.append(float(row["ARGM-ADJ"]))
            list_ARGM_ADV.append(float(row["ARGM-ADV"]))
            list_ARGM_CAU.append(float(row["ARGM-CAU"]))
            list_ARGM_COM.append(float(row["ARGM-COM"]))
            list_ARGM_CXN.append(float(row["ARGM-CXN"]))
            list_ARGM_DIR.append(float(row["ARGM-DIR"]))
            list_ARGM_DIS.append(float(row["ARGM-DIS"]))
            list_ARGM_EXT.append(float(row["ARGM-EXT"]))
            list_ARGM_GOL.append(float(row["ARGM-GOL"]))
            list_ARGM_LOC.append(float(row["ARGM-LOC"]))
            list_ARGM_LVB.append(float(row["ARGM-LVB"]))
            list_ARGM_MNR.append(float(row["ARGM-MNR"]))
            list_ARGM_MOD.append(float(row["ARGM-MOD"]))
            list_ARGM_NEG.append(float(row["ARGM-NEG"]))
            list_ARGM_PRD.append(float(row["ARGM-PRD"]))
            list_ARGM_PRP.append(float(row["ARGM-PRP"]))
            list_ARGM_PRR.append(float(row["ARGM-PRR"]))
            list_ARGM_REC.append(float(row["ARGM-REC"]))
            list_ARGM_TMP.append(float(row["ARGM-TMP"]))
            list_C_ARG0.append(float(row["C-ARG0"]))
            list_C_ARG1.append(float(row["C-ARG1"]))
            list_C_ARG1_DSP.append(float(row["C-ARG1-DSP"]))
            list_C_ARG2.append(float(row["C-ARG2"]))
            list_C_ARG3.append(float(row["C-ARG3"]))
            list_C_ARG4.append(float(row["C-ARG4"]))
            list_C_ARGM_ADV.append(float(row["C-ARGM-ADV"]))
            list_C_ARGM_COM.append(float(row["C-ARGM-COM"]))
            list_C_ARGM_CXN.append(float(row["C-ARGM-CXN"]))
            list_C_ARGM_DIR.append(float(row["C-ARGM-DIR"]))
            list_C_ARGM_EXT.append(float(row["C-ARGM-EXT"]))
            list_C_ARGM_GOL.append(float(row["C-ARGM-GOL"]))
            list_C_ARGM_LOC.append(float(row["C-ARGM-LOC"]))
            list_C_ARGM_MNR.append(float(row["C-ARGM-MNR"]))
            list_C_ARGM_PRP.append(float(row["C-ARGM-PRP"]))
            list_C_ARGM_PRR.append(float(row["C-ARGM-PRR"]))
            list_C_ARGM_TMP.append(float(row["C-ARGM-TMP"]))
            list_C_V.append(float(row["C-V"]))
            list_R_ARG0.append(float(row["R-ARG0"]))
            list_R_ARG1.append(float(row["R-ARG1"]))
            list_R_ARG2.append(float(row["R-ARG2"]))
            list_R_ARG3.append(float(row["R-ARG3"]))
            list_R_ARG4.append(float(row["R-ARG4"]))
            list_R_ARGM_ADV.append(float(row["R-ARGM-ADV"]))
            list_R_ARGM_CAU.append(float(row["R-ARGM-CAU"]))
            list_R_ARGM_COM.append(float(row["R-ARGM-COM"]))
            list_R_ARGM_DIR.append(float(row["R-ARGM-DIR"]))
            list_R_ARGM_GOL.append(float(row["R-ARGM-GOL"]))
            list_R_ARGM_LOC.append(float(row["R-ARGM-LOC"]))
            list_R_ARGM_MNR.append(float(row["R-ARGM-MNR"]))
            list_R_ARGM_TMP.append(float(row["R-ARGM-TMP"]))

    final_sent_list = [list_ARG0, list_ARG1, list_ARG1_DSP, list_ARG2, list_ARG3, list_ARG4, list_ARG5, list_ARGA,
                       list_ARGM_ADJ, list_ARGM_ADV, list_ARGM_CAU, list_ARGM_COM, list_ARGM_CXN, list_ARGM_DIR,
                       list_ARGM_DIS, list_ARGM_EXT, list_ARGM_GOL, list_ARGM_LOC, list_ARGM_LVB, list_ARGM_MNR,
                       list_ARGM_MOD, list_ARGM_NEG, list_ARGM_PRD, list_ARGM_PRP, list_ARGM_PRR, list_ARGM_REC,
                       list_ARGM_TMP, list_C_ARG0, list_C_ARG1, list_C_ARG1_DSP, list_C_ARG2, list_C_ARG3, list_C_ARG4,
                       list_C_ARGM_ADV, list_C_ARGM_COM, list_C_ARGM_CXN, list_C_ARGM_DIR, list_C_ARGM_EXT,
                       list_C_ARGM_GOL, list_C_ARGM_LOC, list_C_ARGM_MNR, list_C_ARGM_PRP, list_C_ARGM_PRR,
                       list_C_ARGM_TMP, list_C_V, list_R_ARG0, list_R_ARG1, list_R_ARG2, list_R_ARG3, list_R_ARG4,
                       list_R_ARGM_ADV, list_R_ARGM_CAU, list_R_ARGM_COM, list_R_ARGM_DIR, list_R_ARGM_GOL,
                       list_R_ARGM_LOC, list_R_ARGM_MNR, list_R_ARGM_TMP]

    best_combination = find_best_combination(final_sent_list, class_names)
