import re
import pickle
import spacy

nonContent = re.compile(r"[=( \n]")

nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe("sentencizer")


def wikiCleaner(wikiPage):
    cleanPage = ""
    wikiPageList = wikiPage.split("\n")
    for line in wikiPageList:
        if not (re.match(nonContent, line) or not line):
            cleanPage += line.strip()
    return cleanPage


def tokenize(corpus_File):

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
    print(corpus)
    print(length)

