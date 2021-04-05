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
import re
# from bmi.search.index import BasicParser


class BasicParser:
    @staticmethod
    def parse(text):
        return re.findall(r"[^\W\d_]+|\d+", text.lower())


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
            heapq.heappush(self.ranking, (score, docid))
            self.status += 1
        else:
            item = heapq.heappop(self.ranking)
            if item[0] < score:
                heapq.heappush(self.ranking, (score, docid))
            else:
                heapq.heappush(self.ranking, item)

    def __iter__(self):
        # min_l = min(len(self.ranking), self.cutoff)
        # sort ranking
        # self.ranking.sort(key=lambda tup: tup[1], reverse=True)
        return iter([(item[1], item[0]) for item in heapq.nlargest(self.cutoff, self.ranking)])


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
                idfs = idf(self.index.doc_freq(term), self.index.ndocs())
                if posting[0] in dic:
                    dic[posting[0]] += tfs * idfs
                else:
                    dic[posting[0]] = tfs * idfs
        for docid in dic:
            ranking.push(self.index.doc_path(docid),
                         dic[docid] / self.index.doc_module(docid))
        return ranking


class DocBasedVSMSearcher(Searcher):

    def __init__(self, index, parser=BasicParser()):
        super().__init__(index, parser)

    def search(self, query, cutoff):
        qterms = self.parser.parse(query)
        ranking = SearchRanking(cutoff)
        heap = list()
        heapq.heapify(heap)
        postings_list = {}

        for term in qterms:
            # Guardamos para cada termino sus postings y la posicion actual
            postings_list[term] = [item for item in self.index.postings(term)]

            # Almacenamos en el heap la tripleta (docid ,score, termino) del primer docid de los postings
            posting = postings_list[term].pop(0)
            heapq.heappush(heap, (posting[0], self.score(posting[1], term), term))

        id = -1
        sc = 0
        while len(heap) > 0:
            item = heapq.heappop(heap)
            if id != item[0]:
                if id  != -1:
                    ranking.push(self.index.doc_path(id),sc/self.index.doc_module(id))
                id = item[0]
                sc = 0

            sc += item[1]

            if len(postings_list[item[2]]) > 0:
                posting = postings_list[item[2]].pop(0)
                heapq.heappush(heap, (posting[0], self.score(posting[1], item[2]), item[2]))

        ranking.push(self.index.doc_path(id),sc/self.index.doc_module(id))
        return ranking

    def score(self, freq, term):
        return tf(freq) * idf(self.index.doc_freq(term), self.index.ndocs())



class ProximitySearcher(Searcher):
    def __init__(self, index, parser=BasicParser()):
        super().__init__(index, parser)

    def search(self, query, cutoff):
        qterms = self.parser.parse(query)
        posiciones = {}
        docids = set()
        ranking = SearchRanking(cutoff)
        ini = 1
        flag = 0
        # Guardamos los postings posicionales y los docids de los documentos con
        # score distinto de 0
        for term in qterms:

            posiciones[term] = dict(self.index.positional_postings(term))
            if ini :
                docids = set(posiciones[term].keys())
                ini = 0
            else:
                docids = docids.intersection(set(posiciones[term].keys()))


        for id in docids:
            score = 0
            p = []
            max_list = []
            for j in range(len(qterms)):
                p.append(0)
                max_list.append(posiciones[qterms[j]][id][0])
            b = max(max_list)

            while b != -1:
                i = 0
                for j in range(len(qterms)):
                    if(len(posiciones[qterms[j]][id]) <= (p[j] + 1)):
                        if posiciones[qterms[j]][id][p[j]] < posiciones[qterms[i]][id][p[i]]:
                            i = j
                        continue

                    while posiciones[qterms[j]][id][p[j]+1] <= b:
                        p[j]+=1
                        if(len(posiciones[qterms[j]][id]) <= (p[j]+1)):
                            break

                    if posiciones[qterms[j]][id][p[j]] < posiciones[qterms[i]][id][p[i]]:
                        i = j

                a = posiciones[qterms[i]][id][p[i]]
                score = score + 1/(b-a-len(qterms)+2)
                p[i] += 1
                if(len(posiciones[qterms[i]][id]) <= p[i]):
                    b = -1
                else:
                    b = posiciones[qterms[i]][id][p[i]]

            ranking.push(self.index.doc_path(id), score)


        return ranking










class PagerankDocScorer():

    def __init__(self, graphfile, r, n_iter):
        # Your new code here (exercise 6) #
        # Format of graphfile:
        #  node1 node2
        out={}
        p={}
        p_prima={}
        links=[]
        with open(graphfile, "r") as f:
            while True:
                linea=f.readline()
                if not linea:
                    break
                nodos=linea.split()
                links.append(nodos)
                if nodos[0] in out:
                    out[nodos[0]] += 1
                else:
                    out[nodos[0]]=1

                if nodos[1] not in out:
                    out[nodos[1]]=0

        N=len(out)
        for item in out:
            p[item]=1 / N

        for d in range(n_iter):
            for i in p:
                p_prima[i]=r / N
            for k in links:
                i=k[0]
                j=k[1]
                p_prima[j]=p_prima[j] + (1 - r) * p[i] / out[i]
            for i in p_prima:
                p[i]=p_prima[i]

        self.p=p

    def rank(self, cutoff):
        order_docids=sorted(self.p.items(), key=lambda item: -item[1])

        ret=[]

        for i in range(min(len(order_docids), cutoff)):
            doc=order_docids[i]
            ret.append((doc[0], doc[1]))

        return ret
