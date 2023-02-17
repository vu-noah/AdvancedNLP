# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza
from lxml import etree


def process_text(text):
    """
    Process text with an English stanza pipeline.
    :param str text: a text
    :return: stanza.Document processed_text_stanza, ConstituentTree constituent_tree
    """
    # instantiate stanza pipeline and process text
    stanza_pipeline = stanza.Pipeline('en')
    processed_text_stanza: stanza.Document = stanza_pipeline(text)

    return processed_text_stanza


def parse_string_to_xml(node, constituent_tree_object):
    """
    :param sentence_list:
    :param node:
    :param constituent_tree_object:
    :return:
    """
    for child in constituent_tree_object.children:
        
        #create terminal nodes
        
        if len(str(child).split(' ')) == 2:
            element = etree.SubElement(node, 'terminal')
            element.set('POS', child.label)
            element.text = child.leaf_labels()[0]
            continue
        #create non-terminal nodes
        else:
            try:
                element = etree.SubElement(node, child.label)
            except ValueError:
                element = etree.SubElement(node, 'OTH')
                
        parse_string_to_xml(element, child)

    return node


def add_attributes_to_xml(sentence, tree):
    """

    :param sentence:
    :param tree:
    :return:
    """
    
    head_id_dict = {}
    deprel_dict = {}
    for word in sentence.words:
        
        head_id_dict[word.text] = str(word.head)
        deprel_dict[word.text] = word.deprel
    
    index = 0
    for element in tree.iter():
        if element.tag == 'terminal':
            element.set('index', str(index))
            index += 1
            element.set('head_id', head_id_dict[element.text])
            element.set('deprel', deprel_dict[element.text])
    
    return tree


def get_phrase_type(tree, word):
    """
    :param tree:
    :param word:
    :return:
    """
    for element in tree.iter():
        if element.text == word.text and int(element.get('index')) == word.id-1:
            parent = (element.getparent())

            return parent.tag
        
    
    
def get_whole_constituent(tree, word):
    """

    :param tree:
    :param word:
    :return:
    """
    constituent_tokens = []
    constituent_pos = []
    for element in tree.iter():
        if element.text == word.text and int(element.get('index')) == word.id-1:
            parent = element.getparent()
            whole_constituent = parent.findall(".//terminal")
               
            for element2 in whole_constituent:
                constituent_tokens.append(element2.text.lower())
                constituent_pos.append(element2.get('POS'))
                               
                
    return constituent_tokens, constituent_pos


def extract_features(doc):
    """
    Extract feature dictionaries for every word in a stanza.Document object, store them in lists, zip and return them.
    :param stanza.Document doc: a stanza.Document object containing processed text
    :return: XXX
    """
    # create lists to store feature dictionaries in
    categorical_feature_dictionaries = []
    binary_feature_dictionaries = []

    for sentence in doc.sentences:

        # print(sentence.constituency.pretty_print())

        sentence_token_list = [word.text for word in sentence.words]
        root = etree.Element("sentence")
        tree = parse_string_to_xml(root, sentence.constituency)
        add_attributes_to_xml(sentence, tree)
        # etree.dump(tree)

        
        #for each word in the sentence, map word id to head id
        deprel_dict = {word.id: word.head for word in sentence.words}
        # print(deprel_dict)
        
        for word in sentence.words:

            # create feature dictionaries for the word
            categorical_feature_dictionary = {'word': word.text.lower()}
            numerical_feature_dictionary = {}

            # get the head of the token
            if word.head != 0:
                head_id = word.head-1
                categorical_feature_dictionary['head'] = sentence.words[head_id].text.lower()
            else:
                categorical_feature_dictionary['head'] = word.text.lower()

            # check whether the token is the root
            if word.head == 0:
                numerical_feature_dictionary['is_root'] = 1
            else:
                numerical_feature_dictionary['is_root'] = 0

            # get the phrase type of the phrase the current token belongs to
            categorical_feature_dictionary['phrase_type'] = get_phrase_type(tree, word)

            # get whole constituent (words and POS)
            constituent_tokens, constituent_pos = get_whole_constituent(tree, word)
            categorical_feature_dictionary['constituent_words'] = constituent_tokens
            categorical_feature_dictionary['constituent_POS'] = constituent_pos
                        
            # if NP, governed by what 'S' or 'VP'? + voice (passive/active)
            for element in root.findall('.//terminal'): 
                if element.getparent().tag == 'NP' and element.text == word.text:
                    if element.getparent() not in root.find('.//VP').findall('.//NP'):
                        categorical_feature_dictionary['governed_by'] = 'S'
                        for element in root.findall('.//terminal'):
                            if element.attrib.get('POS') == 'VBN':
                                id = root.findall('.//terminal').index(element)
                                if root.findall('.//terminal')[id-1].text.lower() in ['am','is','are','was','were','been','be']:
                                    categorical_feature_dictionary['government_voice_relation'] = 'governed_by_S_and_verb_passive' 
                    else:
                        categorical_feature_dictionary['governed_by'] = 'VP'
                        for element in root.findall('.//terminal'):
                            if element.attrib.get('POS') == 'VBN':
                                id = root.findall('.//terminal').index(element)
                                if root.findall('.//terminal')[id-1].text.lower() in ['am','is','are','was','were','been','be']:
                                    categorical_feature_dictionary['government_voice_relation'] = 'governed_by_VP_and_verb_passive'

            # get dependency label of the current token 
            categorical_feature_dictionary['dependency_label'] = word.deprel

            #get dependents: tokens, pos_tags, lemmas
            dependents_tokens = []
            dependents_POS = []      
            dependents_lemmas = []
            for word2 in sentence.words:
                if word2.head == word.id: 
                    dependents_tokens.append(word2.text.lower())
                    dependents_POS.append(word2.xpos)
                    dependents_lemmas.append(word2.lemma.lower())
                    categorical_feature_dictionary["dependents_tokens"] = dependents_tokens
                    categorical_feature_dictionary["dependents_POS"] = dependents_POS
                    categorical_feature_dictionary["dependents_lemmas"] = dependents_lemmas

            #get path length
            cur_head_id = deprel_dict[word.id]
            path_length = 0
            while cur_head_id != 0:
                cur_head_id = deprel_dict[cur_head_id]
                path_length += 1
            numerical_feature_dictionary['path_length_to_head'] = path_length

            # append the feature dictionary to the list of feature dictionaries
            categorical_feature_dictionaries.append(categorical_feature_dictionary)
            binary_feature_dictionaries.append(numerical_feature_dictionary)

    return zip(categorical_feature_dictionaries, binary_feature_dictionaries)


def perform_feature_extraction(text):
    """
    Extract feature dictionaries for every token from a text, print the feature dictionaries.
    :param text: a text
    :return: None
    """
    processed_text = process_text(text)
    feature_dictionaries = extract_features(processed_text)
    for categorical_binary_pair in feature_dictionaries:
        print(categorical_binary_pair)


if __name__ == '__main__':
    example_sentence = '''The chubby lama is eating a bunch of grass. Meanwhile, though no-one cared to tell him, a big misunderstanding took place.'''
    perform_feature_extraction(example_sentence)

# '''Everyone has the right to an effective remedy by the competent national tribunals for acts
#     violating the fundamental rights granted him by the constitution or by law.'''
