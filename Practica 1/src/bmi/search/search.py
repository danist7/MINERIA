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


class VSMDotProductSearcher(Searcher):
    hola = None
    ## TODO ##
    # Your code here #

class VSMCosineSearcher(VSMDotProductSearcher):
    hola2 = None
    ## TODO ##
    # Your code here #
