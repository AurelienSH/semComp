"""
Script de vectorisation avec Word2Vec utilisant gensim
"""
from gensim.models import Word2Vec, KeyedVectors
import pickle
import sys

if __name__ == "__main__":
    # Model parameters
    model=Word2Vec(window=5, min_count=2, workers=4, sg=0) # Hyperparamètres du model
    with open("./corpus/pickledTokenizedCorpus", "r+b") as f: # Chargement du corpus
        corpus = pickle.load(f)


    size = int(sys.argv[1]) # Taille d'échantillonnage en nombre de phrases

    corpus = corpus[:size]

    # Entrainement du modèle
    model.build_vocab(corpus, progress_per=3000)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Enregistrement des vecteurs du modèle uniquement
    vectors = model.wv
    vectors.save(f"./models/vectors_w2v_{size}.kv")
