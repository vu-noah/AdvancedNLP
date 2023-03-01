import pandas as pd
import re
import stanza
import json
nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)

def get_information_about_predicate(sent_df):
    """Extracts information about the current predicate. Helper function for extract_features_to_determine_roles
    param sent_df: a Pandas dataframe that represents one sentence
    returns:
        cur_pred_is_head (bool): True if the current predicate is the head of the sentence
        cur_pred_is_passive (bool): True if the current predicate is passive
        cur_pred_id_in_sent (int): the id in the sentence of current predicate
        UPOS_of_cur_pred (string): the UPOS of the current predicate"""
    # we want to know whether the current predicate is the head of the sentence, and whether or not it is passive.
    # we initially assume these to be False
    cur_pred_is_head = False
    cur_pred_is_passive = False

    # the iternum column holds a count that represents which copy of the sentence (and thus, which predicate) we are on
    predicate_iternum = list(sent_df['iternum'])[0]

    # iterate over the predicate column. Our goal is to find the ith predicate for i == predicate_iternum
    counter = 0
    for i, row in sent_df.iterrows():
        # we are only interested in predicates, so we skip any non-predicates (value '_')
        if row['PB_predicate'] == '_':
            continue
        else:
            # the counter value represents one of the predicates in the sentence
            # if this counter equals the iternum, then we are on the row in which the information about the current
            # predicate is stored
            if counter == predicate_iternum:
                # we set a variable 'cur_pred_id_in_sent' to represent the predicates id in the sentence, and
                # 'UPOS_of_cur_pred' to represent the predicate's UPOS
                cur_pred_id_in_sent = row['token_id_in_sent']
                UPOS_of_cur_pred = row['UPOS']
                # if the predicate is passive, we set cur_pred_is_passive to True
                if "Voice=Pass" in row['grammar']:
                    cur_pred_is_passive = True
                # if the predicate is also the head, we set cur_pred_is_head to True
                if row['head_id'] == 0:
                    cur_pred_is_head = True
                    break
            else:
                counter += 1

    return cur_pred_is_head, cur_pred_is_passive, cur_pred_id_in_sent, UPOS_of_cur_pred


def get_dependency_path(sent_df, row, cur_pred_is_head, cur_pred_id_in_sent):
    """Extracts a dependency path from a token to its dependency head. Helper function for extract_features_to_determine_roles
    param sent_df: a Pandas dataframe that represents one sentence
    param row: the row in the sent_df from which we want to know the path
    param bool cur_pred_is_head: a Bool indicating whether the current predicate is the head
    param int cur_pred_id_in_sent: the index of the current predicate in the sentence

    returns: a list of dependency relations from the current token to the head. If there is no path to the , the list is empty."""

    #create a dictionary that maps token IDs to head IDs
    index_head_dict = {token_id: head_id for token_id, head_id in zip(list(sent_df['token_id_in_sent']),
                                                                          list(sent_df['head_id']))}
    dependency_path_to_pred = []

    # find the id of the current token
    index = row['token_id_in_sent']
    # as long as the head_id of the current token is not the id of the current predicate,
    # we find the head of the current token and add its dependency label to the list 'dependency_path_to_pred'
    while index != cur_pred_id_in_sent:
        # find the current head
        dependency_path_to_pred.append(list(sent_df['dependency_label'])[index-1])
        # if we reach the root (index == 0), we stop the iteration (break)
        if index == 0:
            # if the current predicate isn't the root, that means we have not reached the current
            # predicate, therefore there is no possible dependency path to the predicate. We create an empty path '[]'
            if not cur_pred_is_head:
                dependency_path_to_pred = []
            break
        # if we did not reach the root, we find the next head of the token and continue the while loop
        index = index_head_dict[index]

    return dependency_path_to_pred

def find_governing_constituencies(word, tree):
    """finds the governing constituencies for a given word in a constituency tree. Helper function for extract_features_to_determine_roles.
    Param str word: the token plus its index (e.g. 'token16')
    param str tree: a Stanza constituency tree, parsed to string"""


    bracket_count = 0
    #we will return the strings first_order_constituency_head and second_order_constituency_head
    first_order_constituency_head = ''
    second_order_constituency_head = ''
    #if we have found the first constituency head, we set the following bool to True
    first_head_found = False

    #we check if the word is alphanumerical (re might have difficulty with other characters)
    if word.isalnum():
        #we find the index of the word in the tree
        index = re.search(word, tree).start()
        #we iterate backwards over the characters in the tree, starting from this index
        for i in range(index, 0, -1):
            #we add or subtract 1 from the bracket count based on the brackets we find
            if tree[i] == '(':
                bracket_count += 1
            if tree[i] == ')':
                bracket_count -= 1
            #when we have reached the first order constituency head, the bracket_count is 2
            if bracket_count == 2 and not first_head_found:
                #we look for the tag of the constituency, which is the first string of uppercase characters after the '('
                XP = re.match('[A-Z]+', tree[i+1:])
                #if found, we change first_order_constituency_head to its text
                if XP:
                    first_order_constituency_head = XP.group()
                #we set first_head_found to True: this prevents us from assigning new values
                # to first_order_constituency_head while searching for the second order constituency head
                first_head_found = True
            #when we have reached the first order constituency head, the bracket_count is 2
            if bracket_count == 3:
                #we look for the tag of the constituency, which is the first string of uppercase characters after the '('
                XP = re.match('[A-Z]+', tree[i+1:])
                #if found, we change second_order_constituency_head to its text
                if XP:
                    second_order_constituency_head = XP.group()
                break

    return first_order_constituency_head, second_order_constituency_head

def extract_features_to_determine_roles(filepath):
    """
    Extract features for SR candidate tokens (predicted in the previous step).
    :param str filepath: the path to the file with the (predicted) candidates
    :return: tuple df, categorical_feature_dicts, sentence_level_feature_dicts, numerical_feature_dicts
    """
    # read in the tsv file (that has no header row), assign column names, and store the data in a pandas dataframe
    df = pd.read_csv(filepath, sep='\t', header=0, quotechar='ą', engine='python')  # by setting 'quotechar' to a letter
    # that is not part of the tsv file, we make sure that nothing is counted as a quotechar (to solve the errors with
    # punctuation chars in italics)


    # create three empty lists to put the feature dicts in later
    categorical_feature_dicts = []
    numerical_feature_dicts = []

    # check whether a token has been predicted as a candidate for an SR (if so, the value in 'candidate_prediction' = 1)
    if 'train' in filepath:
        candidate_column = 'is_candidate'
    elif 'test' in filepath:
        candidate_column = 'candidate_prediction'

    # create a dataframe for each sentence (i.e. rows with the same sent_id) in the same order as the original
    # file
    for group in df.groupby('global_sent_id', sort=False):
        sent_df = group[1]

        # A. CREATING SENTENCE LEVEL OBJECTS TO HELP WITH FEATURE EXTRACTION

        # 1. feed the sentence to a stanza pipeline
        sentence = list(sent_df['token'])
        doc = nlp([sentence])
        #create named entities
        entities = doc.sentences[0].ents
        #create a string version of the Stanza parse tree
        tree = doc.sentences[0].constituency
        tree = tree.replace_words([word+str(i+1) for i, word in enumerate(tree.leaf_labels())])
        tree_string = str(tree)
        #count constituents in the tree
        constituent_counter = tree.get_constituent_counts(tree.children)

        # 2. create a counter for the argument candidates, for feature 5)
        argument_count = 0


        # 4. find information about the current predicate (each sentence is copied as many times as there are predicates
        # in the sentence; each copy is linked to a specific (current) predicate for which we want to label the arguments.)
        # we extract whether the current pred is the head (bool), whether it is passive (bool), its id (int), and its UPOS (str)
        cur_pred_is_head, cur_pred_is_passive, cur_pred_id_in_sent, UPOS_of_cur_pred = get_information_about_predicate(sent_df)

        # B. ITERATE OVER TOKENS TO ADD THE FEATURES TO THE FEATURE DICT

        # for each token in the sentence:
        for i, row in sent_df.iterrows():
            if row[candidate_column] == 1:

                # create 2 dicts to store the categorical and numerical features in later
                # we only extract features for tokens that are candidates
                categorical_feature_dict = {}
                numerical_feature_dict = {}


                # 1) extract lemma, PoS, and dependeny label of the current token
                categorical_feature_dict['lemma'] = row['lemma'].lower()
                categorical_feature_dict['UPOS'] = row['UPOS']
                categorical_feature_dict['POS'] = row['POS']
                categorical_feature_dict['dependency_label'] = row['dependency_label']


                # 2) extract lemma and POS of the head
                head_id = row['head_id']
                try:
                    # find row(s) in the dataframe whose token id equals the current token's head id
                    head_lemmas = sent_df.loc[sent_df['token_id_in_sent'] == int(head_id)]
                    categorical_feature_dict['lemma_of_head'] = head_lemmas.iloc[0]['lemma'].lower()
                    categorical_feature_dict['UPOS_of_head'] = head_lemmas.iloc[0]['UPOS']
                    categorical_feature_dict['POS_of_head'] = head_lemmas.iloc[0]['POS']
                    # if the current token is the root, the above gives an IndexError; in that case we add 'None' to
                    # the feature dict
                except IndexError:
                    categorical_feature_dict['lemma_of_head'] = 'token_is_root'
                    categorical_feature_dict['UPOS_of_head'] = 'token_is_root'
                    categorical_feature_dict['POS_of_head'] = 'token_is_root'


                # 3) extract whether the token is a NE (check whether UPOS is PROPN)
                if row['UPOS'] == 'PROPN':
                    numerical_feature_dict['is_NE'] = 1
                else:
                    numerical_feature_dict['is_NE'] = 0


                # 4) obtain voice of the predicate and fill the feature dict 'voice' with the values specified below
                if cur_pred_is_passive:
                    # count += 1
                    categorical_feature_dict['voice'] = 'passive'
                    # categorical_feature_dict['predicate_order'] = f'{count}_passive'
                else:
                    categorical_feature_dict['voice'] = 'active'
                    # categorical_feature_dict['predicate_order'] = '_'

                #5) get argument order in combination with whether the predicate is a passive verb, active verb or other
                # each argument candidate in the sentence gets an updated value for argument_count
                argument_count += 1
                # check if the current predicate is a verb
                if UPOS_of_cur_pred in ['AUX', 'VERB']:
                    #c heck whether it is passive or active
                    if cur_pred_is_passive:
                        categorical_feature_dict['argument_order_voice'] = f"{argument_count}_passive"
                    else:
                        categorical_feature_dict['argument_order_voice'] = f"{argument_count}_active"
                # if the predicate is not a verb, assign it the category 'other'
                else:
                    categorical_feature_dict['argument_order_voice'] = f"{argument_count}_other"


                # 6) get the distance to the current predicate
                cur_index = row['token_id_in_sent']
                distance = cur_index - cur_pred_id_in_sent
                numerical_feature_dict['distance_to_predicate'] = distance


                # 7) binary feature to determine whether the token is before or after predicate
                if distance < 0:
                    # token is before the predicate
                    numerical_feature_dict['before_predicate'] = 1
                if distance >= 0:
                    # token is after the predicate
                    numerical_feature_dict['before_predicate'] = 0


                # 8) get the NE type of the token
                for ent in entities:
                    if row['token'] == ent.text:
                        categorical_feature_dict['NE_type'] = ent.type
                    else:
                        categorical_feature_dict['NE_type'] = 'O'



                # 9) Get the dependency path from current token to current predicate
                dependency_path_to_pred = get_dependency_path(sent_df, row, cur_pred_is_head, cur_pred_id_in_sent)
                categorical_feature_dict['dependency_path_to_pred'] = dependency_path_to_pred

                # 10) add UPOS of predicate
                categorical_feature_dict['UPOS_of_cur_pred'] = UPOS_of_cur_pred

                # 11) constituency heads
                token = row['token'] + str(row['token_id_in_sent'])
                first_order_head, second_order_head = find_governing_constituencies(token, tree_string)

                categorical_feature_dict['first_order_const_head'] = first_order_head
                categorical_feature_dict['second_order_const_head'] = second_order_head

                # 12 counts of chunks in the sentence
                for chunk, count in constituent_counter.items():
                    numerical_feature_dict[f"{chunk}_count"] = count

                # print(categorical_feature_dict, numerical_feature_dict)
                # append the feature dicts to the list
                categorical_feature_dicts.append(categorical_feature_dict)
                numerical_feature_dicts.append(numerical_feature_dict)

    print('Features extracted.')


    # Writing dicts to .json
    with open("categorical_feature_dicts.json", "w") as outfile:
        outfile.write(json.dumps(categorical_feature_dicts))

    with open("numerical_feature_dicts.json", "w") as outfile:
        outfile.write(json.dumps(numerical_feature_dicts))

    print('dictionaries saved to file.')
    # return the feature dicts and the dataframe


    return df, categorical_feature_dicts, numerical_feature_dicts
