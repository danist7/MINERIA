"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

import whoosh
from whoosh.fields import Schema, TEXT, ID
from whoosh.formats import Format
from whoosh.qparser import QueryParser
from bmi.search.search import Searcher
from bmi.search.index import Index
from bmi.search.index import Builder
from bmi.search.index import FreqVector

# A schema in Whoosh is the set of possible fields in a document in
# the search space. We just define a simple 'Document' schema
Document = Schema(
        ## TODO ##
        # Your code here #
        )

class WhooshBuilder(Builder):
    ## TODO ##
    # Your code here #

class WhooshIndex(Index):
    ## TODO ##
    # Your code here #

class WhooshSearcher(Searcher):
    ## TODO ##
    # Your code here #
