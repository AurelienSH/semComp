"""
Scriot permettant d'extraire des exemples de knn à comparer
"""
import pandas as pd
import sys
from typing import Dict
import glob as glb

def create_comp_from_file(comparison_file: str) -> Dict:
    """
    Fonction permettant d'extraire les colonnes à comparer à partir d'un fichier contenant les informations qu'on veut. Par exemple, la ligne suivante indique que depuis le fichier de KNN pour "ville" (taille d'échantillon 10000), on veut les colonnes de la réduction avec locallyLinearEmbedding, de cette même réduction avec PPMI, et celle avec PPMI et lissage laplacien +1. 
    
    ```plain
    ville_10000 LocallyLinearEmbedding PPMI_LocallyLinearEmbedding PPMI_add1_LocallyLinearEmbedding
    ```
    
    Ca renvoie un dictionnaire qui contient les colonnes. 
    
    Args:
    - comparision_file (str): le chemin vers le fichier contenant les comparaisons qu'on veut 
    
    Returns:
    - comparaisons (dict): le dictionnaire contenant les colonnes demandées
    """
    comparisons = dict()
    with open(comparison_file, "r") as f:
        l = f.readline()
        while l:
            wordSize, *methods = l.split(" ")
            methods[-1] = methods[-1].strip()
            wordSizeTuple = tuple(wordSize.split("_"))
            comparisons[wordSizeTuple] = methods
            l = f.readline()
    return comparisons


def comparisons_to_csv(comparisons: Dict, filename: str) -> pd.DataFrame:
    """
    Fonction permmettant de créer un dataframe à partir d'un dictionnaire contenant les colonnes à comparer. En plus de renvoyer le dataframe, il est enregistré dans un fichier tsv qui porte soit le nom spécifié en arguments, soit un nom généré.
    
    Args:
    - comparaisons (dict): un dictionnaire contenant les colonnes extraites qu'on veut comparer
    - filename (str): le nom du fichier de sortie (s'il n'est pas mentionné, un nom générique est créé)
    
    Returns:
    DataFrame: le dataframe avec les colonnes qu'on veut comparer
    """
    wordSizes = comparisons.keys()
    folder = "./outfiles/knn/compared/"
    comp_df = pd.DataFrame()
    filename_tmp = ""

    for wordSize in wordSizes:
        word, size = wordSize
        filename_tmp += f"_{word}"
        df = pd.read_csv(f"./outfiles/knn/{word}_{size}.tsv", index_col=0, sep="\t")
        print(f"./outfiles/knn/{word}_{size}.tsv")
        for method in comparisons[wordSize]:
            filename_tmp += f"_{method}"
            print(method + " : "+ str(len(method.split("_"))))
            only_ppmi_check = (method.startswith("PPMI") and len(method.split("_")) < 2)
            print(only_ppmi_check)
            only_ppmi_laplace_check = method.startswith("PPMI_add") and len(method.split("_")) < 3
            print(only_ppmi_laplace_check)
            if only_ppmi_laplace_check or only_ppmi_check:
                comp_df[f"{word}_{size}_{method}"] = df[f"{method}"]
            else:
                for full_method in df.keys():
                    if full_method.startswith(method):
                        comp_df[f"{word}_{size}_{full_method}"] = df[full_method]

    if not filename:
        filename = filename_tmp

    comp_df.to_csv(f"{folder}/{filename}", sep="\t")
    return comp_df


def comparison_file_to_csv(comparison_file: str, filename: str = None) -> None:
    """
    Fonction permettant de partir d'un fichier de liste de colonnes à comparer à l'écriture d'un fichier csv de la Table voulue. Concaténation des fonctions `create_comp_from_file()` et `comparisons_to_csv()`.
    
    Args:
    - comparison_file (str): le fichier contenant les colonnes qu'on veut comparer
    - filename (str): le nom du fichier dans lequel on veut enregistrer le csv (par défaut, None)
    """
    comparisons = create_comp_from_file(comparison_file)
    comparisons_to_csv(comparisons, filename)
    return None


if __name__ == "__main__":
    comparison_folder = "./input_files_for_comparison"
    comparison_files = [file for file in glb.glob(f"{comparison_folder}/*") if file.split(".")[-1]=="txt"]
    for comparison_file in comparison_files:
        comparison_file_to_csv(comparison_file, f"{comparison_file.split('/')[-1].split('.')[0]}.tsv")
