"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

import math
import heapq
from abc import ABC, abstractmethod
from bmi.search.index import BasicParser

def tf(freq):
    return 1 + math.log2(freq) if freq > 0 else 0

def idf(df, n):
    return math.log2((n + 1) / (df + 0.5))

class SearchRanking:
    # TODO: to be implemented as heap (exercise 1.3) #
    def __init__(self, cutoff):
        self.ranking = list()
        heapq.heapify(self.ranking)
        self.cutoff = cutoff
        self.status = 0

    def push(self, docid, score):
        if self.status < self.cutoff:
            heapq.heappush(self.ranking,(score,docid))
            self.status += 1
        else:
            item = heapq.heappop(self.ranking)
            if item[0] < score:
                heapq.heappush(self.ranking,(score,docid))
            else:
                heapq.heappush(self.ranking,item)

    def __iter__(self):
        #min_l = min(len(self.ranking), self.cutoff)
        ## sort ranking
        #self.ranking.sort(key=lambda tup: tup[1], reverse=True)
        return iter([(item[1],item[0]) for item in heapq.nlargest(self.cutoff,self.ranking)])

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

    # TODO ver que hacer con el parser
    def __init__(self, index, parser=BasicParser()):
        super().__init__(index, parser)

    def search(self, query, cutoff):
        dic = {}
        qterms = self.parser.parse(query)
        ranking = SearchRanking(cutoff)
        for term in qterms:
            for posting in self.index.postings(term):
                tfs = tf(posting[1])
                idfs = idf(self.index.doc_freq(term),self.index.ndocs())
                if posting[0] in dic:
                    dic[posting[0]] += tfs*idfs
                else:
                    dic[posting[0]] = tfs*idfs
        for docid in dic:
            ranking.push(self.index.doc_path(docid), dic[docid]/self.index.doc_module(docid))
        return ranking
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
