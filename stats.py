#############################################
### descriptive statistics about our data ###
#############################################

### imports
import os
from os import listdir
from os.path import join
import math


###########################
### basic preprocessing ###
###########################

def read_text(filename):
    """Read text and return string with full text."""
    with open(filename, 'r') as f:
       text = f.read()
    return text
# --> "A b C t E f G h"


def tokenize(text):
    """Tokenize text by spliting at whitespace."""
    tok_list = text.lower().split()
    return tok_list
# --> ["a", "b", "c", ...]


def count_tokens(tokens):
    tok_freq = dict()
    for index, tok in enumerate(tokens):
        tok_freq[tok] = tok_freq.get(tok, 0) + 1
    return tok_freq



##############################
### higher level functions ###
##############################

def collect_data(directory):
    """
    Read and preprocess all texts.
    Returns a dictionary with keys = filenames
    and values = frequency_dictionaries.
    """
    collection = dict()

    textfiles = os.listdir(directory)
    for f in textfiles:
        path = os.path.join(directory, f)
        text = read_text(path)
        toks = tokenize(text)
        toks_freq = count_tokens(toks)
        collection[f] = toks_freq

    return collection



def tf_idf_general(data):

    # idf for each token
    text_sum = len(list(data.keys()))
    idf = dict()
    for text in data:
        for tok in data[text]:
            idf[tok] = idf.get(tok, 0) + 1
    for tok in idf:
        idf[tok] = round(math.log10(text_sum/idf[tok]), 3)

    # tf for each token in each text
    tf = dict()
    for text in data:
        tok_sum = sum(list(data[text].values()))
        tf[text] = dict()
        for tok in data[text]:
            tf[text][tok] = round(data[text][tok] / tok_sum, 3)

    # tf-idf for each token in each text
    tf_idf = dict()
    for text in tf:
        tf_idf[text] = dict()
        for tok in tf[text]:
            tf_idf[text][tok] = round(tf[text][tok] * idf[tok], 3)
    
    return tf_idf


def reverse_tf_idf_dict(tf_idf_dict):
    """
    Create dictionary with
    - keys = tokens and
    - values = dict[text] -> tf_idf
    """
    
    toks = set()
    for text in tf_idf_dict:
        toks.union(set(tf_idf_dict[text].keys()))
    
    tf_idf_toks = dict()
    for tok in toks:
        tf_idf_toks[tok] = dict()
        for text in tf_idf_dict:
            tf_idf_toks[tok][text] = tf_idf_dict[text].get(tok, 0.0)

    return tf_idf_toks





# boilerplate
if __name__ == '__main__':
    data = collect_data("data")
    tf_idf = tf_idf_general(data)