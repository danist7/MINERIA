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
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os, os.path
import shutil
import pathlib
import zipfile
from whoosh.fields import Schema, TEXT, ID
from whoosh.formats import Format
from whoosh.qparser import QueryParser
from bmi.search.search import Searcher
from bmi.search.index import Index
from bmi.search.index import Builder
from bmi.search.index import TermFreq

# A schema in Whoosh is the set of possible fields in a document in
# the search space. We just define a simple 'Document' schema
Document = Schema(
        path=ID(stored=True),
        title=TEXT(stored=True),
        content=TEXT(vector=Format))


class WhooshBuilder(Builder):
    def __init__(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.writer = whoosh.index.create_in(path, Document).writer()
        return

    def build(self, collection_path):
        # Caso carpeta
        try:
            for path in pathlib.Path(collection_path).iterdir():
                path = "./" + str(path)
                self.writer.add_document(path=path, content=BeautifulSoup(open(path, "r").read(), "lxml").text)
            return
        except Exception:
            pass
        # Caso zip
        try:
            with zipfile.ZipFile(collection_path, "r") as f:
                for name in f.namelist():
                    data = f.read(name)
                    self.writer.add_document(path=name, content=BeautifulSoup(data, "lxml").text)
            return
        # Caso txt
        except Exception:
            f = open(collection_path, "r")
            urls = f.readlines()
            for url in urls:
                self.writer.add_document(path=url, content=BeautifulSoup(urlopen(url).read(), "lxml").text)
            f.close()
            return

    def commit(self):
        self.writer.commit()
        return

# Clase que representa un indice
class WhooshIndex(Index):

    # Abre el indice guardado en el directorio dado y crea el reader
    def __init__(self,path):
        self.index = whoosh.index.open_dir(path)
        self.reader = self.index.reader()
        return

    # Devuelve el nº de documentos que contienen el termino dado
    def doc_freq(self, term):
        return self.reader.doc_frequency("content", term)

    # Devuelve una lista con todos los términos del indice
    def all_terms(self):
        list = []
        for t in self.reader.all_terms():
            if str(t[0]) == "content":
                list.append(t[1].decode("utf-8"))

        return list

    # Devuelve una lista de tuplas, una por cada termino del indice
    # Las tuplas son de la forma (termino, frecuencia total en el indice)
    def all_terms_with_freq(self):
        list = []
        terms = self.all_terms()
        # Por cada termino calcula la frecuencia total
        for t in terms:
            freq = self.total_freq(t)
            list.append((t, freq))

        return list

    # Devuelve la frecuencia total deun termino en el indice
    # TODO: Revisar
    def total_freq(self, term):
        return self.reader.frequency("content", term)

    # Devuelve un lista de elementos de tipo TermFreq, que contienen tuplas de
    # la forma (termino, frecuencia en el documento)
    def doc_vector(self, doc_id):
        list = []
        vec = self.reader.vector(doc_id, "content").items_as("frequency")
        for t in vec:
            list.append(TermFreq(t))
        return list

    # Devuelve elpath de un documento
    def doc_path(self, doc_id):
        return self.reader.stored_fields(doc_id)['path']

    # Devuelve la frecuencia de un termino en un documento
    def term_freq(self, term, doc_id):
        vec = self.reader.vector(doc_id, "content")
        vec.skip_to(term)
        return vec.value_as("frequency")

    # Devuelve una lista de tuplas de la forma
    # (id del documento, frecuencia del termino)
    def postings(self, term):
        list = []
        for doc in self.reader.all_doc_ids():
            freq = self.term_freq(term,doc)
            list.append((doc,freq))
        return list


class WhooshSearcher(Searcher):

    def __init__(self,path):
        index = whoosh.index.open_dir(path)
        super().__init__(index,QueryParser("content", schema=index.schema))
        return

    def search(self, query, cutoff):
        list = []
        for docid, score in self.index.searcher().search(self.parser.parse(query)).items():
            list.append((self.index.reader().stored_fields(docid)['path'],score))
        return list
