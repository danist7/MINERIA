"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""
import bmi.search.search as s
import os
import os.path
import shutil
import pickle
import zipfile
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import sys
import math


class Config(object):
    # variables de clase
    NORMS_FILE = "docnorms.dat"
    PATHS_FILE = "docpaths.dat"
    INDEX_FILE = "serialindex.dat"
    DICTIONARY_FILE = "dictionary.dat"
    POSTINGS_FILE = "postings.dat"


class BasicParser:
    @staticmethod
    def parse(text):
        return re.findall(r"[^\W\d_]+|\d+", text.lower())


class TermFreq():
    def __init__(self, t):
        self.info = t

    def term(self):
        return self.info[0]

    def freq(self):
        return self.info[1]


class Index:
    def __init__(self, dir=None):
        self.docmap = []
        self.modulemap = {}
        if dir:
            self.open(dir)

    def add_doc(self, path):
        self.docmap.append(path)  # Assumed to come in order

    def doc_path(self, docid):
        return self.docmap[docid]

    def doc_module(self, docid):
        if docid in self.modulemap:
            return self.modulemap[docid]
        return None

    def ndocs(self):
        return len(self.docmap)

    def doc_freq(self, term):
        return len(self.postings(term))

    def term_freq(self, term, docID):
        post = self.postings(term)
        if post is None:
            return 0
        for posting in post:
            if posting[0] == docID:
                return posting[1]
        return 0

    def total_freq(self, term):
        freq = 0
        for posting in self.postings(term):
            freq += posting[1]
        return freq

    def doc_vector(self, docID):
        # used in forward indices
        return list()

    def postings(self, term):
        # used in more efficient implementations
        return list()

    def positional_postings(self, term):
        # used in positional implementations
        return list()

    def all_terms(self):
        return list()

    def save(self, dir):
        if not self.modulemap:
            self.compute_modules()
        p = os.path.join(dir, Config.NORMS_FILE)
        with open(p, 'wb') as f:
            pickle.dump(self.modulemap, f)

    def open(self, dir):
        try:
            p = os.path.join(dir, Config.NORMS_FILE)
            with open(p, 'rb') as f:
                self.modulemap = pickle.load(f)
        except OSError:
            # the file may not exist the first time
            pass

    def compute_modules(self):
        for term in self.all_terms():
            idf = s.idf(self.doc_freq(term), self.ndocs())
            post = self.postings(term)
            if post is None:
                continue
            for docid, freq in post:
                if docid not in self.modulemap:
                    self.modulemap[docid] = 0
                self.modulemap[docid] += math.pow(s.tf(freq) * idf, 2)
        for docid in range(self.ndocs()):
            self.modulemap[docid] = math.sqrt(
                self.modulemap[docid]) if docid in self.modulemap else 0


class Builder:
    def __init__(self, dir, parser=BasicParser()):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        self.parser = parser

    def build(self, path):
        if zipfile.is_zipfile(path):
            self.index_zip(path)
        elif os.path.isdir(path):
            self.index_dir(path)
        else:
            self.index_url_file(path)

    def index_zip(self, filename):
        file = zipfile.ZipFile(
            filename, mode='r', compression=zipfile.ZIP_DEFLATED)
        for name in file.namelist():
            with file.open(name, "r", force_zip64=True) as f:
                self.index_document(name, BeautifulSoup(
                    f.read().decode("utf-8"), "html.parser").text)
        file.close()

    def index_dir(self, dir):
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                path = os.path.join(dir, file)
                with open(path, "r") as f:
                    self.index_document(path, f.read())

    def index_url_file(self, file):
        with open(file, "r") as f:
            self.index_urls(line.rstrip('\n') for line in f)

    def index_urls(self, urls):
        for url in urls:
            self.index_document(url, BeautifulSoup(
                urlopen(url).read().decode("utf-8"), "html.parser").text)

    def index_document(self, path, text):
        pass

    def commit(self):
        pass


class RAMIndex(Index):
    def __init__(self, dir=None):
        self.modulemap = {}
        self.docmap = []
        self.diccionario = set()
        self.postingsdict = {}
        super().save(dir)
        if not dir :
            return
        # Abrimos el fichero de los paths
        p = os.path.join(dir, Config.PATHS_FILE)
        file = open(p,'rb')
        self.docmap = pickle.load(file)
        file.close()
        # Abrimos el fichero del diccionario
        p = os.path.join(dir, Config.DICTIONARY_FILE)
        file = open(p,'rb')
        self.diccionario = pickle.load(file)
        file.close()
        # Abrimos el fichero del postings
        p = os.path.join(dir, Config.POSTINGS_FILE)
        file = open(p,'rb')
        self.postingsdict = pickle.load(file)
        file.close()

    def postings(self, term):
        lista = self.postingsdict[term]
        return list(zip(lista[0::2],lista[1::2]))

    def all_terms(self):
        return list(self.diccionario)

class RAMIndexBuilder(Builder):
    def __init__(self, dir, parser=BasicParser()):
        super().__init__(dir, parser)
        self.dir = dir
        self.id = 0
        self.paths = []
        self.diccionario = set()
        self.postings = {}


    def index_document(self, path, text):
        # Guardamos el id y el path del documento
        self.paths.append(path)
        # Para cada termino en el texto
        for t in self.parser.parse(text):
            # Aniadimos los terminos al diccionario
            self.diccionario.add(t)
            # Creamos los postings
            # Si el termino esta en los postings
            if t in self.postings:
                # Si ya ha aparecido el termino en el documento
                if self.postings[t][-2] == self.id:
                    # Aniadimos uno al contador
                    self.postings[t][-1] += 1
                # Si no ha aparecido
                else:
                    # Aniadimos el posting y lo ponemos a 1
                    self.postings[t].append(self.id)
                    self.postings[t].append(1)
            # Si el termino no esta en los postings
            else:
                # Creamos la lista de postings y guardamos el documento
                self.postings[t] = []
                self.postings[t].append(self.id)
                self.postings[t].append(1)
        # Aumentamos el id
        self.id += 1

    def commit(self):
        # Guardamos en disco los paths
        p = os.path.join(self.dir, Config.PATHS_FILE)
        if os.path.exists(p):
            os.remove(p)
        file = open(p, "wb")
        pickle.dump(self.paths, file)
        file.close()
        # Guardamos en disco los diccionarios
        p = os.path.join(self.dir, Config.DICTIONARY_FILE)
        if os.path.exists(p):
            os.remove(p)
        file = open(p, "wb")
        pickle.dump(self.diccionario, file)
        file.close()
        # Guardamos en disco los postings
        p = os.path.join(self.dir, Config.POSTINGS_FILE)
        if os.path.exists(p):
            os.remove(p)
        file = open(p, "wb")
        pickle.dump(self.postings, file)
        file.close()


class DiskIndex(Index):
    # Your new code here (exercise 3*) #
    pass


class DiskIndexBuilder(Builder):
    # Your new code here (exercise 3*) #
    pass


class PositionalIndex(Index):
    # Your new code here (exercise 5*) #
    # Note that it may be better to inherit from a different class
    # if your index extends a particular type of index
    # For example: PositionalIndex(RAMIndex)
    pass


"""
class PositionalIndexBuilder(IndexBuilder):
    # Your new code here (exercise 5*) #
    # Same note as for PositionalIndex
    pass"""
