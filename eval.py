"""
Script permettant d'obtenir les knn d'un mots selon sa méthode de vectorisation
"""

from sklearn.neighbors import NearestNeighbors
import sys
import pandas as pd
import glob as glb
import numpy as np
from gensim.models.word2vec import KeyedVectors, Word2Vec
from typing import Tuple, List


def w2v_knn(size: int,target: str) -> np.array :
    """
    fonction récupérant les knn d'un mot parmi les vecteurs d'un modèle Word2Vec
    """
    w2v_knn = []
    wv = KeyedVectors.load(f'./models/vectors_w2v_{size}.kv')
    normal_neighbors = wv.most_similar([target], topn=10)
    for neighbor, sim in normal_neighbors:
        w2v_knn.append(neighbor)
    w2v_knn = np.array(w2v_knn)
    return w2v_knn

def knn(matrix: pd.DataFrame,target: str) -> np.array(str):
    """
    Fonction permettant d'obtenir les knn d'un mot à partir d'un dataframe de vecteurs
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

def make_path_and_name(matrix: pd.DataFrame) -> Tuple(str):
    """
    Fonction de création du chemin d'enregistrement du csv de sortie
    """
    size, method, param, paramAtt = matrix.split(" ")
    path = f"./outfiles/{size}_sentences/{method}/{param}/{method}_{param}_{paramAtt}"
    name = f"{method}_{param}_{paramAtt}"
    return path, name

def create_name_from_path(path: str) -> str:
    """
    Fonction permettant d'obtenir le nom d'une méthode et des paramètres à partir de son chemin
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
            name = f"{method}_{param}_{paramAtt.split('.')[0]}"
    elif len(filename.split("_")) == 4 :
        _, _, param, paramAtt = filename.split("_")
        name = f"PPMI_{method}_{param}_{paramAtt.split('.')[0]}"
    else:
        _, laplace, _, param, paramAtt = filename.split("_")
        name = f"PPMI_{laplace}_{method}_{param}_{paramAtt.split('.')[0]}"

    return name
def create_neighbors_file(target_word: str,matrices: List[str],size: str, folder = True) -> None:
    """
    fonction créant le fichier des knn d'un mot
    """
    allKnn = dict()
    file_name = f"./outfiles/knn/{target_word}"
    for m in matrices:
        if not folder:
            path, name = make_path_and_name(m)
        else:
            path = m
            name = create_name_from_path(path)
        df = pd.read_csv(path, index_col=0, sep="\t")
        allKnn[name] = knn(df, target_word)
    allKnn["w2v"] = w2v_knn(size, target_word)
    knn_df = pd.DataFrame.from_dict(allKnn)
    knn_df.to_csv(f"{file_name}_{size}.tsv", sep="\t")

    return None

if __name__ == "__main__":
    target_word = "./25mots.txt"
    # size = sys.argv[2]
    with open()

    # if len(sys.argv) > 2:
    #     matrices = []
    #     with open(sys.argv[2],"r") as f:
    #         matrix = f.readline().strip()
    #         while matrix:
    #             matrices.append(matrix)
    #             matrix = f.readline().strip()
    # else:
    #     matrices = glb.glob("./outfiles/100_sentences/*/*/*")
    #
    # if matrices == 'all_largest':
    #     methods = []
    # else:

    # Récupération de toutes les matrices d'une taille d'échantillonnage
    matrices = glb.glob(f"./outfiles/{size}_sentences/*/*")
    create_neighbors_file(target_word,matrices,size)
