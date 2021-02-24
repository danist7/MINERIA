from bmi.search.whooshy import WhooshBuilder, WhooshIndex, WhooshSearcher
import matplotlib.pyplot as plt
import os
import shutil


def term_stats(index):
    freq_dic = {}
    doc_dic = {}
    for term in index.all_terms():

        freq = index.total_freq(term)
        freq_dic[term] = freq

        doc_freq = index.doc_freq(term)
        doc_dic[term] = doc_freq

    order_freq = sorted(freq_dic.items(), key=lambda item: -item[1])
    order_doc = sorted(doc_dic.items(), key=lambda item: -item[1])

    return order_freq, order_doc


def clear(index_path: str):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    else:
        print("Creating " + index_path)
    os.makedirs(index_path)


def toy():
    # Ponemos el lugar donde esta la coleccion y el indice
    index_root_dir = "./index/"
    collections_root_dir = "./data/collections/"
    index_path = index_root_dir + "toy"
    collection = collections_root_dir + "toy"

    # Creamos el indice
    clear(index_path)
    builder = WhooshBuilder(index_path)
    builder.build(collection)
    builder.commit()
    index = WhooshIndex(index_path)
    # Obtenemos la frecuencia de palabras y de documentos
    freq, doc = term_stats(index)
    # De cada lista obtenemos los valores y hacemos plot
    values = [item[1] for item in freq]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Frecuencia de la palabra')
    ax.set_xlabel('Palabra')
    ax.set_title('Frecuencias de las palabras respecto a cada palabra')

    # plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_temrs_toy.jpeg", bbox_inches='tight')

    values = [item[1] for item in doc]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Documentos en los que la palabra aparece')
    ax.set_xlabel('Palabra')
    ax.set_title(
        'Frecuencias de los documentos en los que aparece cada palabra respecto a cada palabra')

    #plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_docs_toy.jpeg", bbox_inches='tight')
    return

def urls():
    # Ponemos el lugar donde esta la coleccion y el indice
    index_root_dir = "./index/"
    collections_root_dir = "./data/collections/"
    index_path = index_root_dir + "urls"
    collection = collections_root_dir + "urls.txt"

    # Creamos el indice
    clear(index_path)
    builder = WhooshBuilder(index_path)
    builder.build(collection)
    builder.commit()
    index = WhooshIndex(index_path)
    # Obtenemos la frecuencia de palabras y de documentos
    freq, doc = term_stats(index)
    # De cada lista obtenemos los valores y hacemos plot
    values = [item[1] for item in freq]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Frecuencia de la palabra')
    ax.set_xlabel('Palabra')
    ax.set_title('Frecuencias de las palabras respecto a cada palabra')

    #plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_temrs_urls.jpeg", bbox_inches='tight')

    values = [item[1] for item in doc]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Documentos en los que la palabra aparece')
    ax.set_xlabel('Palabra')
    ax.set_title(
        'Frecuencias de los documentos en los que aparece cada palabra respecto a cada palabra')

    #plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_docs_urls.jpeg", bbox_inches='tight')
    return

def docs1k():
    # Ponemos el lugar donde esta la coleccion y el indice
    index_root_dir = "./index/"
    collections_root_dir = "./data/collections/"
    index_path = index_root_dir + "docs"
    collection = collections_root_dir + "docs1k.zip"

    # Creamos el indice
    clear(index_path)
    builder = WhooshBuilder(index_path)
    builder.build(collection)
    builder.commit()
    index = WhooshIndex(index_path)
    # Obtenemos la frecuencia de palabras y de documentos
    freq, doc = term_stats(index)
    # De cada lista obtenemos los valores y hacemos plot
    values = [item[1] for item in freq]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Frecuencia de la palabra')
    ax.set_xlabel('Palabra')
    ax.set_title('Frecuencias de las palabras respecto a cada palabra')

    #plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_temrs_docs1k.jpeg", bbox_inches='tight')

    values = [item[1] for item in doc]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.plot(range(len(freq)), values, 'o-', color='r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Documentos en los que la palabra aparece')
    ax.set_xlabel('Palabra')
    ax.set_title(
        'Frecuencias de los documentos en los que aparece cada palabra respecto a cada palabra')

    #plt.xticks(range(len(freq)), labels=labels, rotation='vertical')
    plt.savefig("./memoria/freq_docs_docs1k.jpeg", bbox_inches='tight')
    return

def main():
    toy()
    urls()
    docs1k()
    return

main()
