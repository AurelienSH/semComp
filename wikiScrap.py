"""
Script de création du corpus utilisé pour le projet.
"""

# ============================================================= #

import wikipedia
from traitements import wikiCleaner

# ============================================================= #

wikipedia.set_lang("fr")  # Choix de la langue

pages = wikipedia.random(100000)  # Création d'une liste de 10000 articles wikipedia aléatoires

# Ecriture du corpus
with open("corpus/corpusWiki.txt", "w") as out:
    for page in pages:

        # Utilisation de try/except pour régler une erreur qui survient 
        # quand parfois le titre d'une page n'est pas assez spécifique
        try:
            wikiPage = wikipedia.WikipediaPage(title=page).content  # extraction du contenu textuel d'une page
            cleanPage = wikiCleaner(wikiPage)  # nettoyage du contenu
            print(cleanPage, file=out)  # écriture dans le fichier

        except:
            print("prout")
