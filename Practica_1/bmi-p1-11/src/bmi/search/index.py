"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""
from abc import ABC, abstractmethod


class TermFreq():
    def __init__(self, t):
        self.info = t

    def term(self):
        return self.info[0]

    def freq(self):
        return self.info[1]


class Index(ABC):
    index = None
    reader = None

    @abstractmethod
    def doc_freq(self, term):
        return

    @abstractmethod
    def all_terms(self):
        return

    @abstractmethod
    def all_terms_with_freq(self):
        return

    @abstractmethod
    def total_freq(self, term):
        return

    @abstractmethod
    def doc_vector(self, doc_id):
        return

    @abstractmethod
    def doc_path(self, doc_id):
        return

    @abstractmethod
    def term_freq(self, term, doc_id):
        return

    @abstractmethod
    def postings(self, term):
        return

    @abstractmethod
    def ndocs(self):
        return


class Builder(ABC):
    path = None
    writer = None

    def __init__(self):
        return

    @abstractmethod
    def build(self, collection):
        return

    @abstractmethod
    def commit(self):
        return
