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


"""
    This is an abstract class for the search engines
"""


class Searcher(ABC):
    def __init__(self, index, parser):
        self.index = index
        self.parser = parser

    @abstractmethod
    def search(self, query, cutoff):
        """ Returns a list of documents built as a pair of path and score """

# TODO: Revisar unas cosas de teoria con pablo


class VSMDotProductSearcher(Searcher):
    def __init__(self, index):
        super().__init__(index, None)
        return

    def search(self, query, cutoff):
        # Obtenemos los terminos de la query
        terms = query.lower().split()
        dicterms = {}
        for term in terms:
            if term in dicterms:
                dicterms[term] = dicterms[term] + 1
            else:
                dicterms[term] = 1
        # Diccionario para guardar los docsid y su score
        docids = {}

        for term in terms:
            # Calculamos el idf para el termino
            idf = math.log((self.index.ndocs()+1) / (self.index.doc_freq(term)+0.5))
    
            # Calculamos el tf para cada documento y guardamos el tf*idf
            for posting in self.index.postings(term):
                if posting[1] == 0:
                    tf = 0
                else:
                    tf = 1 + math.log(posting[1])

                if posting[0] in docids:
                    docids[posting[0]] = docids[posting[0]] + tf * idf
                else:
                    docids[posting[0]] = tf * idf

        if cutoff == -1:
            return docids
        # Ordenamos de menor a mayor
        order_docids = sorted(docids.items(), key=lambda item: -item[1])

        ret = []



        for i in range(min(len(order_docids), cutoff)):
            doc = order_docids[i]
            ret.append((self.index.doc_path(doc[0]), doc[1]))
        return ret

# TODO: Pasar al builder lo de los modulos
class VSMCosineSearcher(VSMDotProductSearcher):

    def search(self, query, cutoff):
        prodvec = super().search(query,-1)
        f = open("./index/modulos.txt", "r")
        for id in prodvec:
            modulo = float(f.readline().split()[1])
            prodvec[id] = prodvec[id]/modulo

        order_docids = sorted(prodvec.items(), key=lambda item: -item[1])

        ret = []

        for i in range(min(len(order_docids), cutoff)):
            doc = order_docids[i]
            ret.append((self.index.doc_path(doc[0]), doc[1]))
        return ret
