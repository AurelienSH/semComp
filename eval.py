"""
Script permettant d'obtenir les knn d'un mots selon sa méthode de vectorisation
"""

# ============================================================= #

from sklearn.neighbors import NearestNeighbors
import sys
import pandas as pd
import glob as glb
import numpy as np
from gensim.models.word2vec import KeyedVectors, Word2Vec
from typing import List

# ============================================================= #

def w2v_knn(size: str, target: str) -> np.array :
    """
    Fonction récupérant les knn d'un mot parmi les vecteurs d'un modèle Word2Vec
    
    Args:
    - size (str): la taille de l'échantillon (pour load le bon modèle)
    - target (str): le mot dont on veut trouver les voisins
    
    Returns:
    - np.array: les voisins du mot cible
    """
    w2v_knn = [] # liste des voisins du mot cible
    
    # chargement des vecteurs selon la taille choisie
    wv = KeyedVectors.load(f'./models/vectors_w2v_{size}.kv')
    
    # recherche des voisins
    normal_neighbors = wv.most_similar([target], topn=10)
    
    # on ne garde pas le score de similarité
    for neighbor, _ in normal_neighbors:
        w2v_knn.append(neighbor)
        
    # transformation en np.array
    w2v_knn = np.array(w2v_knn)
    
    return w2v_knn

def knn(matrix: pd.DataFrame, target: str) -> np.array(str):
    """
    Fonction permettant d'obtenir les knn d'un mot à partir d'un dataframe de vecteurs en utilisant la distance cosinus.
    
    Args:
    - matrix (pd.DataFrame): une matrice de co-occurrences
    - target (str): le mot dont on veut trouver les knn parmi la `matrix`
    
    Returns:
    - np.array: les KNN du mot cible
    """

    # Entrainement du modèle de knn de sklearn utilisant la distance cosinus
    neighbors = NearestNeighbors(n_neighbors=11, metric='cosine',algorithm='brute')
    neighbors.fit(matrix)

    # Conversion des données en np.array
    array = (np.array(matrix.loc[target])).reshape(1,-1)

    # Récuperation des indices des knn
    neighbor_index = neighbors.kneighbors(array, return_distance=False)

    # Récupération des clés des vecteurs
    names = list(matrix.index[neighbor_index])

    # Suppression de la ligne trouvant le mot comme son propre voisin
    knn10 = names[0][1:]

    return knn10


def create_name_from_path(path: str) -> str:
    """
    Fonction permettant d'obtenir le nom d'une méthode et ses paramètres à partir de son chemin. Elle sert à bien nommer les colonnes dans les tableaux.
    
    Args:
    - path (str): le chemin vers une matrice
    
    Returns:
    - str: le nom de la colonne pour le fichier de KNN
    """
    _, _, size_sent, method, filename = path.split("/")
    size, _ = size_sent.split("_")
    if len(filename.split("_")) == 2:
        name = f"{method}"
    elif len(filename.split("_")) == 3:
        if method == "PPMI":
            _, laplace, _ = filename.split("_")
            name = f"PPMI_{laplace}"
        else:
            _, param, paramAtt = filename.split("_")
            if method !="VarianceThreshold":
                name = f"{method}_{param}_{paramAtt.split('.')[0]}"
            else:
                name = f"{method}_{param}_{''.join(paramAtt.split('.')[:2])}"
    elif len(filename.split("_")) == 4 :
        _, _, param, paramAtt = filename.split("_")
        if method != "VarianceThreshold":
            name = f"PPMI_{method}_{param}_{paramAtt.split('.')[0]}"
        else:
            name = f"PPMI_{method}_{param}_{''.join(paramAtt.split('.')[:2])}"
    else:
        _, laplace, _, param, paramAtt = filename.split("_")
        if method != "VarianceThreshold":
            name = f"PPMI_{laplace}_{method}_{param}_{paramAtt.split('.')[0]}"
        else:
            name = f"PPMI_{laplace}_{method}_{param}_{''.join(paramAtt.split('.')[:2])}"

    return name

def create_neighbors_file(target_words: List[str], matrices: List[str], size: str) -> None:
    """
    Fonction créant le fichier des knn d'un mot.
    
    Args:
    - target_words (List[str]): liste des mots dont on veut trouver les voisins
    - matrices (List[str]): liste de chemins vers les matrices dans lesquelles on veut trouver les voisins
    - size (str): la taille de l'échantillon à utiliser (pour sélectionner les matrices dans le bon dossier)
    """
    
    allKnn = dict() # initialisation du dico de knn
    
    # pour chaque matrice
    for m in matrices:
        path = m
        name = create_name_from_path(path)
        
        # chargement de la matrice depuis le tsv
        df = pd.read_csv(path, index_col=0, sep="\t")

        # Recherche des knn mot par mot
        for target_word in target_words:
            print(f"Evaluation de la méthode {name} à la taille {size} sur le mot {target_word}")
            allKnn[target_word] = allKnn.get(target_word,dict()) # initialisation du dictionnaire
            # utilisation de try except pour continuer le processus même si le mot n'est pas présent dans les vecteurs
            try:
                allKnn[target_word][name] = knn(df, target_word) # recherche des knn
            except:
                print(f"{target_word} n'est pas dans les vecteurs de {name} à la taille {size}")


    # Ecriture des fichiers à partir du dictionnaire
    for target_word in target_words:
        file_name = f"./outfiles/knn/{target_word}"
        # utilisation de try except pour continuer le processus même si le mot n'est pas présent dans les vecteurs
        try:
            allKnn[target_word]["w2v"] = w2v_knn(size, target_word)
        except:
            print(f"{target_word} n'est pas dans les vecteurs de w2v, à la taille {size}")
        knn_df = pd.DataFrame.from_dict(allKnn[target_word]) # Conversion en dataframe
        knn_df.to_csv(f"{file_name}_{size}.tsv", sep="\t") # Ecriture dans un fichier

    return None

if __name__ == "__main__":
    target_words_file = "./25mots.txt"
    size = sys.argv[1]
    with open(target_words_file, "r") as f:
        target_words = [word.strip() for word in f.readlines()]


    matrices = glb.glob(f"./outfiles/{size}_sentences/*/*") # récupération de tous les fichiers des matrices

    create_neighbors_file(target_words,matrices,size)
