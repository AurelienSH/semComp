"""
Scriot permettant d'extraire des exemples de knn à comparer
"""
import pandas as pd
import sys
from typing import Dict

def create_comp_from_file(comparison_file: str) -> Dict:
    """
    Fonction permettant d'extraire les colonnes à comparer à partir d'un fichier
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
    Fonction permmettant de créer un dataframe à partir d'un dictionnaire contenant les colonnes à comparer
    """
    wordSizes = comparisons.keys()
    folder = "./outfiles/knn/compared/"
    comp_df = pd.DataFrame()
    filename_tmp = ""

    for wordSize in wordSizes:
        word, size = wordSize
        filename_tmp += f"_{word}"
        df = pd.read_csv(f"./outfiles/knn/{word}_{size}.tsv", index_col=0, sep="\t")

        for method in comparisons[wordSize]:
            filename_tmp += f"_{method}"
            only_ppmi_check = method.startswith("PPMI") and len(method.split("_")) < 3
            only_ppmi_laplace_check = method.startswith("PPMI_add") and len(method.split("_")) < 4
            if only_ppmi_laplace_check or only_ppmi_check:
                comp_df[f"{word}_{size}_{method}"] = df[f"{method}_{size}"]
            else:
                for full_method in df.keys():
                    if full_method.startswith(method):
                        comp_df[f"{word}_{size}_{method}"] = df[full_method]

    if not filename:
        filename = filename_tmp

    comp_df.to_csv(f"{folder}/{filename}", sep="\t")
    return comp_df


def comparison_file_to_csv(comparison_file: str, filename=None) -> None:
    """
    Fonction permettant de partir d'un fichier de liste de colonnes à comparer à l'écriture d'un fichier csv de la Table voulue
    """
    comparisons = create_comp_from_file(comparison_file)
    comparisons_to_csv(comparisons, filename)
    return None


if __name__ == "__main__":
    comparison_file = sys.argv[1]
    comparison_file_to_csv(comparison_file)
