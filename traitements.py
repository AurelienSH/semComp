"""
Script de pré-traitement du corpus
"""

# ============================================================= #

# typage des fonctions
from typing import List

# sauvegarde 
import pickle

# nettoyage et traitement
import re
import spacy

# ============================================================= #

nonContent = re.compile(r"[=( \n]") # Expression régulière permettant de se débarrasser des balises entre "="

nlp = spacy.load("fr_core_news_lg") # Utilisation du plus large modèle du français de spacy pour la segmentation du corpus en phrase/tokens
nlp.add_pipe("sentencizer") # Pour la segmentation en phrases


def wikiCleaner(wikiPage: str) -> str:
    """
    Fonction qui nettoie un article Wikipédia. 
    
    Elle prend en entrée le contenu textuel dans un article Wikipédia 
    et renvoie ce même article formaté sur une seule ligne.
    
    Args:
    - wikiPage (str): l'article qu'on veut mettre sur une seule ligne
    
    Returns:
    - cleanPage (str): l'article mis sur une seule ligne
    """
    cleanPage = ""
    wikiPageList = wikiPage.split("\n")
    for line in wikiPageList:
        if not (re.match(nonContent, line) or not line): #si la ligne est autre chose que vide ou du "non contenu"
            cleanPage += line.strip()
    return cleanPage


def tokenize(corpus_File: str) -> List[List[str]]:
    """
    Fonction qui permet d'appliquer les traitements suivants sur un corpus : 
    
    - Segmentation en phrase,
    - Tokenization,
    - Lemmatisation,
    - Suppression des stop-words
    
    Elle prend en entrée un fichier de corpus pré-nettoyé (un article = une ligne, avec la fonction `wikiCleaner()`).
    
    Elle enregistre le corpus traité dans un fichier `pickle`. Elle renvoie également le corpus traité sous forme
    de liste de listes de strings. 
    
    Args:
    - corpus_File (str): le corpus pré-nettoyé avec la fonction `wikiCleaner()`
    
    Returns:
    - tokenizedCorpus (List[List[str]]): le corpus tokenizé, lemmatisé, segmenté en phrases et sans stop words
    """
    tokenizedCorpus = [] # initialisation de la liste qui contiendra chaque phrase du corpus
    
    with open(corpus_File, "r") as corpusF: # ouverture du corpus pré-nettoyé
        
        l = corpusF.readline() # 1 ligne = 1 article
        
        while l:        
            
            doc = nlp(l) # analyse de chaque article par spacy
            
            tokenizedCorpus += (
                [
                    [
                        token.lemma_ for token in sent # on récupère le lemme de chaque token de chaque phrase
                        if not token.is_punct and token.lemma_ != "\n" and not token.is_stop and token.lemma_ # si ce n'est ni une poncutation, ni un retour à la ligne, ni un stop word
                    ]
                    for sent in doc.sents if sent]
            )
            
            l = corpusF.readline()
            
    tokenizedCorpus = [sent for sent in tokenizedCorpus if sent] # suppression des éléments vides 
    
    # sauvegarde du corpus traité avec pickle
    out = open("corpus/pickledTokenizedCorpus", "w+b") 
    pickle.dump(tokenizedCorpus, out)
    out.close()
    
    # on renvoie aussi le corpus traité
    return tokenizedCorpus


if __name__ == "__main__":
    
    # traitement du corpus et enregistrement dans "corpus/pickledTokenizedCorpus"
    tokenize("corpus/corpusWiki.txt")
    
    # ouverture du corpus pickled
    inputFile = open("corpus/pickledTokenizedCorpus", "r+b")
    corpus = pickle.load(inputFile)
    inputFile.close()
    
    # le nombre de tokens dans le corpus
    length = sum([len(sent) for sent in corpus])

