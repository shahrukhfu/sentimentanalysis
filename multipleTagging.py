## Running multiple Taggers for improving effeciency
## Not running correctly


#Hybrid Taggers are used in out code


#Why Taggers Matter?

#Text understanding
#Machine translation
#Speech processing
#Information extraction

from nltk.tag import UnigramTagger,BigramTagger,TrigramTagger,NgramTagger

def tagging(train, taggers, backoff=None):
    # Start with the most general tagger (last in list)
    # and move to most specific (first in list)
    for tagger_class in reversed(taggers):
        backoff = tagger_class(train, backoff=backoff)
    return backoff

## To use -

## patterns=[(r'.*ing','VB'),(r'.*ful','JJ')]
## tagger=tagging(train,[UT,BT,TT],backoff=RegexpTagger(patterns))
