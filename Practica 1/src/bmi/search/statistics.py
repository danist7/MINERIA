from bmi.search.whooshy import WhooshBuilder, WhooshIndex, WhooshSearcher
import matplotlib.pyplot as plt


def term_stats(index):
    freq_dic = {}
    doc_dic = {}
    for term in index.all_terms():

        freq = index.total_freq(term)
        freq_dic[term] = freq

        postings = index.postings(term)
        doc_dic[term] = len(postings)

    order_freq = sorted(freq_dic.items(), key=lambda item: -item[1])
    order_doc = sorted(doc_dic.items(), key=lambda item: -item[1])

    return order_freq, order_doc


def graphic(path, dic, log):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 10, 1])
    ax.plot(range(len(dic.keys())),
            dic.values(), 'o-', color='r')
    #ax.set_ylabel('Número de Alarmas')
    #ax.set_xlabel('Día y Hora')
    #ax.set_title('Serie temporal de Alarmas. Abril.')
    #plt.legend(title='Leyenda')

    plt.xticks(range(len(dic.keys())), labels=dic.keys(), rotation='vertical')
    plt.show()
    plt.savefig(path, bbox_inches='tight')
    return
