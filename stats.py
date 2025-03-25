### imports
import os
from os import listdir
from os.path import join
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import regex as re
import pandas as pd


#############################################
### descriptive statistics about our data ###
#############################################


###########################
### basic preprocessing ###
###########################

def read_text(filename):
    """Read text and return string with full text."""
    with open(filename, 'r', encoding="UTF-8") as f:
       text = f.read().lower()
    return text


def tokenize(text):
    """Tokenize text into word-tokens."""
    text.replace("*", "")
    tok_list = list()
    toks = re.findall(r'\w+\-?\w*', text)
    for index, tok in enumerate(toks):
        if not tok.isnumeric():
            tok_list.append(tok)       
    return tok_list


def sentence_tokens(text):
    """Tokenize text into sentence-tokens."""
    tokenizer = RegexpTokenizer(r'\b[^.\n]+')
    tok_list = tokenizer.tokenize(text)
    return tok_list


def remove_stopwords(token):
    """Remove stopwords from text with the nltk-stopwords-corpus."""    
    stop = nltk.corpus.stopwords.words("german")
    stop.append("für")
    stop.append("nen")
    toks_filtered = [tok for tok in token if tok not in stop]
    return toks_filtered


def count_tokens(tokens):
    """Count frequency of tokens in a text."""
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
    and values = frequency_dictionaries
    """
    collection = dict()
    metadata = dict()

    # open textfiles
    textfiles = os.listdir(directory)
    for f in textfiles:
        path = os.path.join(directory, f)
        text = read_text(path)
        
        # collect metadata for each text
        metadata[f] = dict()
        toks = tokenize(text)
        metadata[f]["count_words"] = len(toks)
        metadata[f]["count_wordtypes"] = len(set(toks))

        toks = remove_stopwords(toks)
        metadata[f]["count_toks"] = len(toks)
        metadata[f]["count_toktypes"] = len(set(toks))

        toks_sent = sentence_tokens(text)
        metadata[f]["count_sent"] = len(toks_sent)
        
        toks_freq = count_tokens(toks)
        collection[f] = toks_freq

    return collection, metadata


def collect_data_by_journal(directory):
    """
    Read and preprocess all texts, with texts from one newspaper
    being grouped into one text.
    Returns a dictionary with keys = journals
    and values = frequency_dictionaries
    """
    collection = dict()
    metadata = dict()

    # open textfiles
    textfiles = os.listdir(directory)
    for f in textfiles:
        path = os.path.join(directory, f)
        text = read_text(path)

        # join text of the same newspaper in one string
        for char in f:
            if char.isnumeric():
                f = f.replace(char, "")
        collection[f] = collection.get(f, "") + text
        
        if f not in metadata.keys():
            metadata[f] = dict()
        metadata[f]["count_articles"] = metadata[f].get("count_articles", 0) + 1

    # collect metadata for each newspaper
    for journal in collection:
        metadata[journal]["count_sent"] = len(sentence_tokens(collection[journal]))

        collection[journal] = tokenize(collection[journal])
        metadata[journal]["count_words"] = len(collection[journal])
        metadata[journal]["count_wordtypes"] = len(set(collection[journal]))

        collection[journal] = remove_stopwords(collection[journal])
        metadata[journal]["count_toks"] = len(collection[journal])
        metadata[journal]["count_toktypes"] = len(set(collection[journal]))
        
        collection[journal] = count_tokens(collection[journal]) 

    return collection, metadata


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


def top_freq(data: dict, x=50):
    """Create frequency-dictionary with 50 most frequent tokens."""
    top_freqs = dict() 
    for key in data:
        tok_freqs = list(data[key].items())
        tok_freqs.sort(key=lambda x: x[1], reverse=True)
        top_freqs[key] = tok_freqs[:x]
    return top_freqs


# boilerplate
if __name__ == '__main__':
    pass