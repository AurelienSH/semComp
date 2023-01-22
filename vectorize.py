"""
Script et module de vectorisation en matrice Terme-Terme de co-occurents
"""
import numpy as np
import pandas as pd
import pickle
import sys
from typing import List
def vectorize(corpus: List[List[str]], size=10000, window=5, ppmi=False, laplace=None, save=True):
    """
    Fonction permettant à partir d'un corpus segmenté en phrase et tokenisé
    de renvoyer un dataframe de la matrice de co-occurences du corpus
    """
    vocab = set() #initialisation du vocabulaire
    corpus = corpus[:size] #echantillonage du corpus

    # Ecriture du vocabulaire
    for sentence in corpus:
        vocab.update(sentence)

    # Conversion du vocabulaire en liste pour obtenir les clés du dataframe
    terms_l = list(vocab)
    cooccurrence_matrix = pd.DataFrame(0, index=terms_l, columns=terms_l) #initialisation de la matrice
    print("Début de vectorisation...")
    for sentence in corpus:
        for i, term1 in enumerate(sentence):
            for j in range(i+1, min(i+window, len(sentence))): # définition de la fenêtre de co-occurence
                term2 = sentence[j]
                if term1!=term2: # exclusion du terme comme co-ocurrent de lui-même
                    cooccurrence_matrix.loc[term1][term2] += 1
                    cooccurrence_matrix.loc[term2][term1] += 1
    print("fini matrice")
    print(f"avant ppmi et laplace, taille : {cooccurrence_matrix.shape}")

    cooccurrence_matrix = cooccurrence_matrix.fillna(0)
    print("fini fillna")

    # lissage laplacien
    if laplace:
        cooccurrence_matrix = cooccurrence_matrix.add(laplace)
        cooccurrence_matrix = ppmi_(cooccurrence_matrix) # Comme le lissage ne fait sens que dans le contexte d'une ppmi, il n'est pas nécessaire de vérifier, l'erreur serait justifiée
        if save:
            cooccurrence_matrix.to_csv(f"./outfiles/{size}_sentences/PPMI/PPMI_add{laplace}_{size}.tsv", sep='\t',encoding='utf-8')
        cooccurrence_matrix = cooccurrence_matrix.fillna(0)
        return cooccurrence_matrix

    # simple application de ppmi
    if ppmi:
        cooccurrence_matrix = ppmi_(cooccurrence_matrix)
        if save:
            cooccurrence_matrix.to_csv(f"./outfiles/{size}_sentences/PPMI/PPMI_{size}.tsv", sep='\t', encoding='utf-8')
        cooccurrence_matrix = cooccurrence_matrix.fillna(0)
        return cooccurrence_matrix

    # Cas dans lequel aucun traitement n'est fait sur la matrice
    if save:
        cooccurrence_matrix.to_csv(f"./outfiles/{size}_sentences/simpleCount/simpleCount_{size}.tsv", sep='\t', encoding='utf-8')
    return cooccurrence_matrix

def ppmi_(df):
    """
    Fonction permettant de calculer efficacement la ppmi depuis un dataframe
    auteur : Pascal Potvin (https://gist.github.com/TheLoneNut)
    source : https://gist.github.com/TheLoneNut/208cd69bbca7cd7c53af26470581ec1e
    """
    # Transformation du dataframe en np array
    arr = np.array(df)

    # Calcul de la probabilité conditionnelle
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T

    # calcul de la probabilité de l'événement
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    # normalisation des données (valeurs alalnt de 0 à 1)
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001

    # application de la ppmi
    _pmi = np.log(ratio)
    _pmi[_pmi < 0] = 0 # conservation uniquement des valeurs positives

    # application de la fonction au dataframe
    pmi_df = pd.DataFrame(_pmi, columns=df.columns, index=df.index)
    return pmi_df


def pickle_load(filename):
    with open(filename, "r+b") as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    size = int(sys.argv[1])
    corpus = pickle_load("corpus/pickledTokenizedCorpus")
    corpus = corpus[:size]
    vectorized_corpus = vectorize(corpus)
