"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

# Flat whoosh example

import whoosh
from whoosh.fields import Schema, TEXT, ID
from whoosh.formats import Format
from whoosh.qparser import QueryParser
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os, os.path
import shutil

Document = Schema(
        path=ID(stored=True),
        title=TEXT(stored=True),
        content=TEXT(vector=Format))

def build_index(dir, urls):
    if os.path.exists(dir): shutil.rmtree(dir)
    os.makedirs(dir)
    writer = whoosh.index.create_in(dir, Document).writer()
    for url in urls:
        writer.add_document(path=url, content=BeautifulSoup(urlopen(url).read(), "lxml").text)
    writer.commit()

def search(dir, query):
    index = whoosh.index.open_dir(dir)
    searcher = index.searcher()
    qparser = QueryParser("content", schema=index.schema)
    print("Search results for '", query, "'")
    for docid, score in searcher.search(qparser.parse(query)).items():
        print(score, "\t", index.reader().stored_fields(docid)['path'])
    print()

def examine(dir, term, docid, n):
    reader = whoosh.index.open_dir(dir).reader()
    print("Total nº of documents in the collection:", reader.doc_count())
    print("Total frequency of '", term, "':", reader.frequency("content", term))
    print("Nº documents containing '", term, "':", reader.doc_frequency("content", term))
    raw_vec = reader.vector(docid, "content")
    raw_vec.skip_to("probability")
    print("Frequency of '", raw_vec.id(), "' in document", docid, reader.stored_fields(docid)['path'], ":", raw_vec.value_as("frequency"))
    print("Top", n, "most frequent terms in document", docid, reader.stored_fields(docid)['path']) 
    vec = reader.vector(docid, "content").items_as("frequency")
    for p in sorted(vec, key=lambda x: x[1], reverse=True)[0:n]:
        print("\t", p)
    print()

urls = ["https://en.wikipedia.org/wiki/Simpson's_paradox", 
        "https://en.wikipedia.org/wiki/Bias",
        "https://en.wikipedia.org/wiki/Entropy"]

dir = "index/whoosh/example/urls"

build_index(dir, urls)

search(dir, "probability")

examine(dir, "probability", 0, 5)
