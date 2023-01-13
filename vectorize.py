import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import pickle


def make_vocab(corpus):
    vocab = set()
    for sent in corpus:
        for word in sent:
            vocab.add(word)
    return vocab


def find_all_indices(sent, target):
    indices = []
    for i, w in enumerate(sent):
        if w == target:
            indices.append(i)
    return indices


def words_in_window(word, index, sent, window):
    words = []
    start = max((index - window), 0)
    end = min((index + window) + 1, len(sent))
    i = start
    while i < end:
        if sent[i] != word:
            words.append(sent[i])
        i+=1
    return words


def vectorize(corpus, window=5):
    vocab = make_vocab(corpus)
    vectorized_corpus = defaultdict(Counter)


    for i,sent in enumerate(corpus):
        print(f"{i/len(corpus)}% done")
        for word in vocab:
            if word in sent:
                indices = find_all_indices(sent, word)
                for index in indices:
                    context = words_in_window(word, index, sent, window)
                    for cooc in context:
                        if cooc!=word:
                            vectorized_corpus[word][cooc] += 1

    vectorized_corpus = dict(vectorized_corpus)
    for key in vectorized_corpus.keys():
        newDict = dict(vectorized_corpus[key])
        vectorized_corpus[key] = newDict

    vectorized_corpus = pd.DataFrame(vectorized_corpus)
    vectorized_corpus = vectorized_corpus.fillna(0)

    return vectorized_corpus


def vectorize_to_csv(corpus, filename, window=5):
    vectorized_corpus = vectorize(corpus, window)
    vectorized_corpus.to_csv(filename, sep='\t', encoding='utf-8')
    return vectorized_corpus


def pickle_load(filename):
    with open(filename, "r+b") as f:
        obj = pickle.load(f)
    return obj

def pickle_dump


if __name__ == "__main__":
    corpus = pickle_load("corpus/pickledTokenizedCorpus")
    vectorize_to_csv(corpus, "matrice.csv")
