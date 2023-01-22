"""
Script de vectorisation avec Word2Vec utilisant gensim
"""

# ============================================================= #

# vectorisation
from gensim.models import Word2Vec, KeyedVectors

# sauvegarde
import pickle

# récupération des arguments en ligne de commande
import sys

# ============================================================= #

if __name__ == "__main__":
    
    # paramétrage du modèle word2vec
    model = Word2Vec(window=5, min_count=2, workers=4, sg=0)
    
    # Chargement du corpus depuis le fichier pickled
    with open("./corpus/pickledTokenizedCorpus", "r+b") as f: 
        corpus = pickle.load(f)

    size = int(sys.argv[1]) # Taille d'échantillonnage en nombre de phrases
    corpus = corpus[:size] # échantillonnage du corpus

    # Entrainement du modèle
    model.build_vocab(corpus, progress_per=3000)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Enregistrement des vecteurs du modèle uniquement
    vectors = model.wv
    vectors.save(f"./models/vectors_w2v_{size}.kv")
