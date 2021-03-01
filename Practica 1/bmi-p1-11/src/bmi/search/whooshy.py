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
import os
import os.path
import shutil
import pathlib
import zipfile
from bmi.search.search import Searcher
from bmi.search.index import Index
from bmi.search.index import Builder
from bmi.search.index import TermFreq
import math

# A schema in Whoosh is the set of possible fields in a document in
# the search space. We just define a simple 'Document' schema
Document = Schema(
    path=ID(stored=True),
    title=TEXT(stored=True),
    content=TEXT(vector=Format))

# Clase que genera un constructor para un indice WhooshSearcher


class WhooshBuilder(Builder):
    def __init__(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.writer = whoosh.index.create_in(path, Document).writer()
        self.path = path
        return

    # Crea el indice para la coleccion
    def build(self, collection_path):
        # Caso carpeta
        try:
            for path in pathlib.Path(collection_path).iterdir():
                path = "./" + str(path)
                self.writer.add_document(path=path, content=BeautifulSoup(
                    open(path, "r").read(), "lxml").text)
            return
        except Exception:
            pass
        # Caso zip
        try:
            with zipfile.ZipFile(collection_path, "r") as f:
                for name in f.namelist():
                    data = f.read(name)
                    self.writer.add_document(
                        path=name, content=BeautifulSoup(data, "lxml").text)
            return
        # Caso txt
        except Exception:
            f = open(collection_path, "r")
            urls = f.readlines()
            for url in urls:
                self.writer.add_document(path=url, content=BeautifulSoup(
                    urlopen(url).read(), "lxml").text)
            f.close()
            return

    # Hace commit a los documentos dek indice y crea un archivo
    # con los modulos de cada uno de los documentos
    def commit(self):
        # Creamos el archivo
        f = open("./index/modulos.txt", "w")
        # Hacemos commit
        self.writer.commit()
        index = WhooshIndex(self.path)
        docs = index.all_doc_ids()
        # Para cada documento obtenemos sus terminos y calculamos su tf idf
        for id in docs:
            modulo = 0
            terms = index.doc_vector(id)
            for TermFreq in terms:
                idf = math.log((index.ndocs() + 1) /
                               (index.doc_freq(TermFreq.term()) + 0.5))
                if TermFreq.freq() == 0:
                    tf = 0
                else:
                    tf = 1 + math.log(TermFreq.freq())
                modulo = modulo + pow(idf * tf, 2)

            modulo = math.sqrt(modulo)
            f.write(str(id) + " " + str(modulo) + '\n')
        f.close()
        return

# Clase que representa un indice


class WhooshIndex(Index):

    # Abre el indice guardado en el directorio dado y crea el reader
    def __init__(self, path):
        self.index = whoosh.index.open_dir(path)
        self.reader = self.index.reader()
        return

    # Devuelve el nº de documentos que contienen el termino dado
    def doc_freq(self, term):
        return self.reader.doc_frequency("content", term)

    # Devuelve el número total de documentos
    def ndocs(self):
        return self.reader.doc_count()

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

    # Devuelve la frecuencia total de un termino en el indice
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

    # Devuelve el path de un documento
    def doc_path(self, doc_id):
        return self.reader.stored_fields(doc_id)['path']

    # Devuelve la frecuencia de un termino en un documento
    def term_freq(self, term, doc_id):
        vec = self.reader.vector(doc_id, "content")
        vec.skip_to(term)
        if(vec.id() != term):
            return 0
        else:
            return vec.value_as("frequency")

    # Devuelve una lista de tuplas de la forma
    # (id del documento, frecuencia del termino)
    def postings(self, term):
        list = []
        for doc in self.reader.all_doc_ids():
            freq = self.term_freq(term, doc)
            list.append((doc, freq))
        return list

    # Devuelve un alista de todos los ids de documentos del indice
    def all_doc_ids(self):
        return self.reader.all_doc_ids()


# Clase que representa el buscador de Whoosh

class WhooshSearcher(Searcher):

    def __init__(self, path):
        index = whoosh.index.open_dir(path)
        super().__init__(index, QueryParser("content", schema=index.schema))
        return

    # Ejecuta la búsqueda de una query dada a través del searcher del index
    # de Whoosh.Devuelve una lista de tamaño cutoff de los documentos y su score
    def search(self, query, cutoff):
        list = []
        for docid, score in self.index.searcher().search(self.parser.parse(query)).items():
            list.append(
                (self.index.reader().stored_fields(docid)['path'], score))

        if cutoff < len(list):
            return list[:cutoff]

        return list
