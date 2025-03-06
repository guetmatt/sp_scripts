#############################################
### descriptive statistics about our data ###
#############################################


"""
TO ADD:
    - (pos-tagging)
    - (stemming)
"""

### imports
import os
from os import listdir
from os.path import join
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


###########################
### basic preprocessing ###
###########################

def read_text(filename):
    """Read text and return string with full text."""
    with open(filename, 'r', encoding="UTF-8") as f:
       text = f.read().lower()
    return text
# --> "A b C t E f G h"


def tokenize(text):
    """Tokenize text into word-tokens."""
    # tok_list = text.lower().split()
    # return tok_list
    tokenizer = RegexpTokenizer(r'\w+\-\w+|\w+')
    tok_list = tokenizer.tokenize(text)
    return tok_list
# --> ["a", "b", "c", ...]


def sentence_tokens(text):
    """Tokenize text into sentence-tokens."""
    tokenizer = RegexpTokenizer(r'\b[^.\n]+')
    tok_list = tokenizer.tokenize(text)
    return tok_list


def remove_stopwords(token):
    """Remove stopwords from text."""    
    stop = nltk.corpus.stopwords.words("german")
    toks_filtered = [tok for tok in token if tok not in stop]
    return toks_filtered


def count_tokens(tokens):
    tok_freq = dict()
    for index, tok in enumerate(tokens):
        tok_freq[tok] = tok_freq.get(tok, 0) + 1
    return tok_freq



def count_quotations(text):
    """Count of "-characters divided by 2 as estimation of quotations."""
    return text.count('"')




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
        toks = remove_stopwords(toks)
        toks_sent = sentence_tokens(text)
        toks_freq = count_tokens(toks)
        collection[f] = toks_freq

    print(toks[:50])
    print(toks_sent[:5])
    return collection



def tf_idf_general(data):
    """
    Caluclate tfidf-score for each token in each text.
    Returns dictionary with
    - keys = texts
    - values = dict[text] -> dict[text][tok] = tfidf
    """

    # idf for each token
    text_sum = len(list(data.keys()))
    idf = dict()
    for text in data:
        for tok in data[text]:
            idf[tok] = idf.get(tok, 0) + 1
    for tok in idf:
        idf[tok] = round(math.log10(text_sum/idf[tok]), 15)

    # tf for each token in each text
    tf = dict()
    for text in data:
        tok_sum = sum(list(data[text].values()))
        tf[text] = dict()
        for tok in data[text]:
            tf[text][tok] = round(data[text][tok] / tok_sum, 15)

    # tf-idf for each token in each text
    tf_idf = dict()
    for text in tf:
        tf_idf[text] = dict()
        for tok in tf[text]:
            tf_idf[text][tok] = round(tf[text][tok] * idf[tok], 15)
    
    return tf_idf


def reverse_tf_idf_dict(tfidf_dict):
    """
    Create dictionary with
    - keys = tokens and
    - values = dict[text] -> tf_idf
    """
    
    toks = set()
    for text in tfidf_dict:
        toks.union(set(tfidf_dict[text].keys()))
    
    tfidf_toks = dict()
    for tok in toks:
        tfidf_toks[tok] = dict()
        for text in tfidf_dict:
            tfidf_toks[tok][text] = tfidf_dict[text].get(tok, 0.0)

    return tfidf_toks


def top_tfidf(tfidf_dict, x=10):
    """
    Create dictionary with
    - keys = texts
    - values = list with tuples of x tokens with highest tfidf-scores [(token, tfidf), (token2, tfidf2), ...]
    """
    tfidf_sorted = dict()
    for text in tfidf_dict:
        tfidf_list = list(tfidf_dict[text].items())
        tfidf_list.sort(reverse=True, key=lambda x: x[1])

        tfidf_sorted[text] = tfidf_list[:x]
    
    return tfidf_sorted




# boilerplate
if __name__ == '__main__':
    data = collect_data("data")
    # tf_idf = tf_idf_general(data)

    # top = top_tfidf(tf_idf, x=20)

    # print()
    # count = 1
    # for entry in top["zeit3.txt"]:
    #     print(f"{count}\t{entry}")
    #     count += 1
    # print()



    # for text in top:
    #     print(text)
    #     for entry in top[text]:
    #         print("\t", entry)
    #     print()
    #     print()