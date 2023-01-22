"""
Script de réduction de dimensionalité
"""

# ============================================================= #

# méthodes de réduction de dimensionalité
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import LocallyLinearEmbedding, MDS
import vectorize as v

# autres
import pandas as pd
from collections import namedtuple
from os import path
import os
import sys
from typing import List

# ============================================================= #²

# Création d'un namedtuple permettant de manipuler plus simplement les matrices
reducedMatrix = namedtuple("reducedMatrix", ["name", "matrix", "path"])

# liste de méthodes de réduction de dimensionalité utilisées avec leurs paramètres
methods = [
    #(PCA, {'n_components': 100}),
    #(NMF, {'n_components': 100}),
    #(LocallyLinearEmbedding, {'n_components': 100}),
    #(MDS, {'n_components': 100}),
    #(VarianceThreshold, {'threshold': 0.3})
]

def get_name(method, params: dict) -> str:
    """
    Fonction permettant d'obtenir le nom d'une méthode de réduction de dimensionalité 
    sous le format `nom_params`. Elle est utile pour le nommage des fichiers de sortie. 
    
    Par exemple `(PCA, {'n_components': 100})` aura comme nom `PCA_ncomponents_100`
    
    Args: 
    - method: la méthode dont on veut obtenir le nom formaté avec ses paramètres
    - params (dict): le dictionnaire des hyperapramètres de la méthode dont on veut obtenir le nom formaté
    
    Returns:
    - pathStr (str): le nom formaté sous la format `nom_params`
    """
    name = method.__name__ # récupération du nom de la méthode
    
    # pour "n_components", on enlève le _ parce que plus tard on split sur les _ 
    if 'n_components' in params.keys():
        param_str = f"ncomponents_{params['n_components']}"
    elif 'threshold' in params.keys():
        param_str = f"threshold_{params['threshold']}"
    return f"{name}_{param_str}"

def get_path(name: str) -> str:
    """
    Méthode permettant d'obtenir le chemin dans lequel la matrice devra être enregistrée.
    
    Agrs:
    - name (str): le nom complet d'une méthode (issu de `get_name`)
    
    Returns:
    - pathStr (str): le chemin dans lequel la matrice doit être enregistrée
    """
    global size # taille de l'échantillon
    pathList = name.split("_") 
    pathStr = f"./outfiles/{size}_sentences"
    for dir in pathList[:-2]:
        pathStr = f"{pathStr}/{dir}"
        if not path.exists(pathStr):
            os.makedirs(pathStr)
    return pathStr


def reduceFeatures(methods: List[tuple], matrix: pd.DataFrame, ppmi: bool = False, laplace: int = None) -> List[reducedMatrix]:
    """
    Fonction qui applique sur la matrice de co-coccurrences `matrix` chaque méthode de réduction de dimensionalité
    qui se trouve dans la liste de tuples `methods` (tuples du type `(method, params)`). 
    
    On peut également si on le veut appliquer une PPMI et un lissage laplacien.  
    
    Chaque matrice réduite est enregistrée dans un fichier bien nommé. 
    
    Args:
    - methods (List[tuple]): une liste de tuples des méthodes de réduction de dimensionalité qu'on veut appliquer `(method, params)`
    - matrix (pd.DataFrame): la matrice de co-occurrences 
    - ppmi (bool): si True, ajoute la mention `pppi` dans le nom du fichier (par défault False)
    - laplace (int): si spécifié, mentionne la valeur utilisée précédée de `add_` dans le nom du fichier (par défaut None)
    
    Returns:
    - reducedMatrices (List[reducedMatrix]): liste de tuples nommés reducedMatrix qui correspond au nom du fichier et au chemin dans lequel est enregistré la matrice et l'objet matrice lui-même
    """
    
    reducedMatrices = [] # intialisation de la liste des matrices réduites

    for method, params in methods:
        # Initialisation de certaines données nécessaires pour une bonne sauvegarde
        method_name = get_name(method,params)
        method_path = get_path(method_name)
        
        # concaténation de la méthode et de ses paramètres
        model = method(**params)

        # Réduction de dimensionalité
        trans = model.fit_transform(matrix)
        
        # conversion en dataframe
        method_matrix = pd.DataFrame(trans, index=matrix.index)
        
        # création du tuple nommé reducedMatric avec le nom de la méthode, la matrice et le chemin d'enregistrement
        rMatrix = reducedMatrix(name=method_name, matrix=method_matrix, path=method_path)
        
        # affichage de la taille de la matrice réduite
        print(f"Taille obtenue pour {rMatrix.name}",rMatrix.matrix.shape)
        
        
        reducedMatrices.append(rMatrix)

        # Bonne nomenclature pour le nommage des fichiers
        if not ppmi:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/{rMatrix.name}.tsv", sep="\t")
        elif not laplace:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/PPMI_{rMatrix.name}.tsv", sep="\t")
        else:
            rMatrix.matrix.to_csv(f"{rMatrix.path}/PPMI_add{laplace}_{rMatrix.name}.tsv", sep="\t")
            
    return reducedMatrices


if __name__ == "__main__":
    
    # chargement du corpus depuis le fichier pickled
    corpus = v.pickle_load("./corpus/pickledTokenizedCorpus")
    size = int(sys.argv[1]) # taille de l'échantillon

    # Différents cas de figure selon l'utilisation de laplace, de la ppmi, ou non
    
    # ppmi et lissage laplacien
    if len(sys.argv) > 3:
        ppmi = sys.argv[2]
        laplace = int(sys.argv[3])
        matrix = v.vectorize(corpus,size=size, ppmi=ppmi,laplace=laplace)
        print("Taille originale",matrix.shape)
        
        # liste des matrices réduites avec chacune des méthodes
        reducedMatrices = reduceFeatures(methods,matrix,ppmi=ppmi,laplace=laplace)
        
    # ppmi sans lissage
    elif len(sys.argv) > 2: 
        ppmi = sys.argv[2]
        matrix = v.vectorize(corpus, size=size, ppmi=ppmi, save=True)
        print("Taille originale", matrix.shape)
        reducedMatrices = reduceFeatures(methods, matrix, ppmi=ppmi)
    
    # ni ppmi ni lissage
    else:
        matrix = v.vectorize(corpus, size=size)
        print("Taille originale", matrix.shape)
        reducedMatrices = reduceFeatures(methods, matrix)


