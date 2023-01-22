"""
Script et module de vectorisation en matrice Terme-Terme de co-occurents
"""

# ============================================================= #

import numpy as np
import pandas as pd

# sauvegarde
import pickle

# récupération des arguments en ligne de commande
import sys

# typage des fonctions
from typing import List

# ============================================================= #

def vectorize(corpus: List[List[str]], size: int = 10000, window: int = 5, ppmi: bool = False, laplace: int = None, save: bool = True) -> pd.DataFrame:
    """
    Fonction qui, à partir d'un corpus segmenté en phrases et tokenizé, renvoie un DataFrame de la matrice terme-terme de co-occurrents du corpus.
    
    Args:
    - corpus (List[List[str]]): le corpus tokenizé et segmenté en phrases 
    - size (int): la taille de l'échantillon qu'on veut (par défaut 1000)
    - window (int): la taille de la fenêtre de co-occurrences = le nombre de mot avant et le nombre de mots après (par défaut 5)
    - ppmi (bool): si True, applique une PPMI sur le DataFrame (par défault False)
    - laplace (int): si spécifié, applique un lissage laplacien avec cette valeur (par défaut None)
    - save (bool): si True, sauvegarde la matrice dans `outfiles`
    
    Returns:
    - cooccurrence_matrix (pd.DataFrame): la matrice terme-terme de co-occurrents 
    """
    vocab = set() # initialisation du vocabulaire
    corpus = corpus[:size] # echantillonage du corpus

    # Ecriture du vocabulaire
    for sentence in corpus:
        vocab.update(sentence)

    # Conversion du vocabulaire en liste pour obtenir les clés du dataframe
    terms_l = list(vocab)
    cooccurrence_matrix = pd.DataFrame(0, index=terms_l, columns=terms_l) # initialisation de la matrice
    
    print("Début de vectorisation...")
    
    # pour chaque phrase du corpus
    for sentence in corpus:
        
        for i, term1 in enumerate(sentence):
        
            for j in range(max(i-window, 0), min(i+window, len(sentence))): # définition de la fenêtre de co-occurence
                term2 = sentence[j] # terme co-occurent
        
                if term1!=term2: # exclusion du terme comme co-ocurrent de lui-même
                    
                    # +1 pour chaque co-occurrence dans le datafram
                    cooccurrence_matrix.loc[term1][term2] += 1
                    cooccurrence_matrix.loc[term2][term1] += 1
    
    print("fini matrice")
    print(f"avant ppmi et laplace, taille : {cooccurrence_matrix.shape}") # affichage de la taille de la matrice avant ppmi et laplace

    # on remplace les NaN par des 0
    cooccurrence_matrix = cooccurrence_matrix.fillna(0)
    print("fini fillna")

    # lissage laplacien
    if laplace:
        cooccurrence_matrix = cooccurrence_matrix.add(laplace)
        
        # Comme le lissage ne fait sens que dans le contexte d'une ppmi, il n'est pas nécessaire de vérifier, l'erreur serait justifiée
        cooccurrence_matrix = ppmi_(cooccurrence_matrix) 
        
        cooccurrence_matrix = cooccurrence_matrix.fillna(0) # on remplace les NaN par des 0
        
        # sauvegarde dans `outfile`
        if save:
            cooccurrence_matrix.to_csv(f"./outfiles/{size}_sentences/PPMI/PPMI_add{laplace}_{size}.tsv", sep='\t',encoding='utf-8')
            
        return cooccurrence_matrix

    # simple application de ppmi sans lissage
    if ppmi:
        cooccurrence_matrix = ppmi_(cooccurrence_matrix)
        cooccurrence_matrix = cooccurrence_matrix.fillna(0) # on remplace les NaN par des 0
        
        if save: # sauvegarde dans `outfiles`
            cooccurrence_matrix.to_csv(f"./outfiles/{size}_sentences/PPMI/PPMI_{size}.tsv", sep='\t', encoding='utf-8')
        
        return cooccurrence_matrix

    # Cas dans lequel aucun traitement n'est fait sur la matrice
    if save: # sauvegarde dans `outfiles`
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


def pickle_load(filename: str):
    """Fonction qui permet de charger un corpus depuis un fichier pickled.

    Args:
        filename (str): le chemin vers le fichier pickled qu'on veut charger

    Returns:
        obj: l'objet chargé depuis le fichier pickled
    """
    with open(filename, "r+b") as f:
        obj = pickle.load(f)
    return obj