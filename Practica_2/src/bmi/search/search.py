"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

import math
from abc import ABC, abstractmethod

def tf(freq):
    return 1 + math.log2(freq) if freq > 0 else 0

def idf(df, n):
    return math.log2((n + 1) / (df + 0.5))

class SearchRanking:
    # TODO: to be implemented as heap (exercise 1.3) #
    def __init__(self, cutoff):
        self.ranking = list()
        self.cutoff = cutoff

    def push(self, docid, score):
        self.ranking.append((docid, score))

    def __iter__(self):
        min_l = min(len(self.ranking), self.cutoff)
        ## sort ranking
        self.ranking.sort(key=lambda tup: tup[1], reverse=True)
        return iter(self.ranking[0:min_l])

"""
    This is an abstract class for the search engines
"""
class Searcher(ABC):
    def __init__(self, index, parser):
        self.index = index
        self.parser = parser
    @abstractmethod
    def search(self, query, cutoff) -> SearchRanking:
        """ Returns a list of documents built as a pair of path and score """


class SlowVSMSearcher(Searcher):
    def __init__(self, index, parser=BasicParser()):
        super().__init__(index, parser)

    def search(self, query, cutoff):
        qterms = self.parser.parse(query)
        ranking = SearchRanking(cutoff)
        for docid in range(self.index.ndocs()):
            score = self.score(docid, qterms)
            if score:
                ranking.push(self.index.doc_path(docid), score)
        return ranking

    def score(self, docid, qterms):
        prod = 0
        for term in qterms:
            prod += tf(self.index.term_freq(term, docid)) \
                    * idf(self.index.doc_freq(term), self.index.ndocs())
        mod = self.index.doc_module(docid)
        if mod:
            return prod / mod
        return 0

class TermBasedVSMSearcher(Searcher):
    # Your new code here (exercise 1.1) #
    pass

class DocBasedVSMSearcher(Searcher):
    # Your new code here (exercise 1.2*) #
    pass

class ProximitySearcher(Searcher):
    # Your new code here (exercise 4*) #
    pass

class PagerankDocScorer():
    def __init__(self, graphfile, r, n_iter):
        # Your new code here (exercise 6) #
        # Format of graphfile:
        #  node1 node2
        # TODO #
        pass
    def rank(self, cutoff):
        # TODO #
        pass
