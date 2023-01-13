import wikipedia
from traitements import wikiCleaner

wikipedia.set_lang("fr")

pages = wikipedia.random(100000)

with open("corpus/corpusWiki.txt", "w") as out:

    for page in pages:

        try:
            wikiPage = wikipedia.WikipediaPage(title=page).content
            cleanPage = wikiCleaner(wikiPage)
            print(cleanPage, file=out)

        except:
            print("prout")