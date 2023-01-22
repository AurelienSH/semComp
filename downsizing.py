"""
Script de réduction de dimensionalité
"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import LocallyLinearEmbedding, MDS
import vectorize as v
import pandas as pd
from collections import namedtuple
from os import path
import os
import sys
from typing import List

# Création d'un namedtuple permettant de manipuler plus simplement les matrices
reducedMatrix = namedtuple("reducedMatrix", ["name", "matrix","path"])

# liste de méthodes de réduction de dimensionalité utilisées
methods = [
    (PCA, {'n_components': 100}),
    #(NMF, {'n_components': 100}),
    #(LocallyLinearEmbedding, {'n_components': 100}),
    #(MDS, {'n_components': 100}),
    #(VarianceThreshold, {'threshold': 0.3})

]


def get_name(method: str, params: dict) -> str:
    """
    Fonction permettant d'obtenir le nom détaillé d'une méthode de réduction et ses paramètres
    """
    name = method.__name__
    if 'n_components' in params.keys():
        param_str = f"ncomponents_{params['n_components']}"
    elif 'threshold' in params.keys():
        param_str = f"threshold_{params['threshold']}"
    return f"{name}_{param_str}"

def get_path(name: str) -> str:
    """
    méthode permettant d'obtenir le chemin dans lequel la matrice devra être enregistrée
    """
    global size
    pathList = name.split("_")
    pathStr = f"./outfiles/{size}_sentences"
    for dir in pathList[:-2]:
        pathStr = f"{pathStr}/{dir}"
        if not path.exists(pathStr):
            os.makedirs(pathStr)
    return pathStr


def reduceFeatures(methods: List[tuple], matrix: pd.DataFrame, ppmi=False, laplace=None) -> List[reducedMatrix]:
    """
    fonction de réduction de dimensionalité
    """
    reducedMatrices = []

    for method, params in methods:
        # Initialisation de certaines données nécessaires pour une bonne sauvegarde
        method_name = get_name(method,params)
        method_path = get_path(method_name)
        model = method(**params)

        # Réduction de dimensionalité
        trans = model.fit_transform(matrix)
        method_matrix = pd.DataFrame(trans, index=matrix.index)
        rMatrix = reducedMatrix(name=method_name, matrix=method_matrix, path=method_path)
        print(f"Taille obtenue pour {rMatrix.name}",rMatrix.matrix.shape)
        reducedMatrices.append(rMatrix)

        # Bonne nomenclature
        if not ppmi:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/{rMatrix.name}.tsv", sep="\t")
        elif not laplace:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/PPMI_{rMatrix.name}.tsv", sep="\t")
        else:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/PPMI_add{laplace}_{rMatrix.name}.tsv", sep="\t")
    return reducedMatrices


if __name__ == "__main__":
    corpus = v.pickle_load("./corpus/pickledTokenizedCorpus")
    size = int(sys.argv[1])

    # Différents cas de figure selon l'utilisation de laplace, de la ppmi, ou non
    if len(sys.argv) > 3:
        ppmi = sys.argv[2]
        laplace = int(sys.argv[3])
        matrix = v.vectorize(corpus,size=size, ppmi=ppmi,laplace=laplace)
        print("Taille originale",matrix.shape)
        reducedMatrices = reduceFeatures(methods,matrix,ppmi=ppmi,laplace=laplace)
    elif len(sys.argv) > 2:
        ppmi = sys.argv[2]
        matrix = v.vectorize(corpus, size=size, ppmi=ppmi)
        print("Taille originale", matrix.shape)
        reducedMatrices = reduceFeatures(methods, matrix, ppmi=ppmi)
    else:
        matrix = v.vectorize(corpus, size=size)
        print("Taille originale", matrix.shape)
        reducedMatrices = reduceFeatures(methods, matrix)


