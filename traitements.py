"""
Script de pré-traitement du corpus
"""
from typing import List
import re
import pickle
import spacy

nonContent = re.compile(r"[=( \n]") # Expression régulière permettant de se débarrasser des balises entre "="

nlp = spacy.load("fr_core_news_lg") # Utilisation du plus large modèle du français de spacy pour la segmentation du corpus en phrase/tokens
nlp.add_pipe("sentencizer")


def wikiCleaner(wikiPage: str) -> str:
    """
    Fonction ayant pour but de formatter chaque articule du corpus sur une ligne
    """
    cleanPage = ""
    wikiPageList = wikiPage.split("\n")
    for line in wikiPageList:
        if not (re.match(nonContent, line) or not line): #si la ligne est autre chose que vide ou du "non contenu"
            cleanPage += line.strip()
    return cleanPage


def tokenize(corpus_File: str) -> List[List[str]]:
    """
    Fonction prenant un fichier de corpus pré-nettoyé en entrée et renvoyant un corpus
    segmenté en phrases, tokenisé, lemmatisé et sans stop-words
    et qui le conserve dans un fichier pickle
    """
    tokenizedCorpus = []
    with open(corpus_File, "r") as corpusF:
        l = corpusF.readline()
        while l:
            doc = nlp(l)
            tokenizedCorpus += (
                [
                    [
                        token.lemma_ for token in sent
                        if not token.is_punct and token.lemma_ != "\n" and not token.is_stop and token.lemma_
                    ]
                    for sent in doc.sents if sent]
            )
            l = corpusF.readline()
    tokenizedCorpus = [sent for sent in tokenizedCorpus if sent]
    out = open("corpus/pickledTokenizedCorpus", "w+b")
    pickle.dump(tokenizedCorpus, out)
    out.close()
    return tokenizedCorpus


if __name__ == "__main__":
    tokenize("corpus/corpusWiki.txt")
    inputFile = open("corpus/pickledTokenizedCorpus", "r+b")
    corpus = pickle.load(inputFile)
    inputFile.close()
    length = sum([len(sent) for sent in corpus])

