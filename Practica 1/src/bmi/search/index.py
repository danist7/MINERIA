"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

class TermFreq():
    def __init__(self, t):
        self.info = t
    def term(self):
        return self.info[0]
    def freq(self):
        return self.info[1]


class Index:
    index = None
    reader = None
    def __init__(self):
        return

    def doc_freq(self, term):
        reader = index.reader();
        return
    def all_terms(self):
        return
    def all_terms_with_freq(self):
        # devuelve tupla
        return
    def total_freq(self, term):
        return
    def doc_vector(self, doc_id):
        # tupla término/frecuencia en TermFreq
        return
    def doc_path(self, doc_id):
        return
    def term_freq(self, term, doc_id):
        return
    def postings(self, term):
        # lista de tuplas documento/ frecuencia
        return




class Builder:
    writer = None
    def __init__(self):
        return
    def build(self, collection):
        return
    def commit(self):
        return
