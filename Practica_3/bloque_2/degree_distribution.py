import networkx as nx
from sna import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as sc
import time
import datetime





def main():
    dir = "graph/"
    #network = transforma_csv(dir,"small1.csv","small1_transformado.csv")
    #G = estudiar_grafo(dir+"small1_transformado.csv","small1",dir)
#    paradoja_amistad(network,grado_medio(G),"small1")

    #network= transforma_csv(dir,"small2.csv","small2_transformado.csv")
    #G  = estudiar_grafo(dir+"small2_transformado.csv","small2",dir)
    #paradoja_amistad(network,grado_medio(G),"small2")

#    network= transforma_csv(dir,"small3.csv","small3_transformado.csv")
    #G = estudiar_grafo(dir+"small3_transformado.csv","small3",dir)
    #paradoja_amistad(network,grado_medio(G),"small3")

    #network = transforma_csv(dir,"Erdös-Rényi.csv","Erdös-Rényi_transformado.csv")
    #G = estudiar_grafo(dir+"Erdös-Rényi_transformado.csv","Erdös-Rényi",dir)
    #dibujar_grafo(G,dir,"Erdös-Rényi")
    #paradoja_amistad(network,grado_medio(G),"Erdös-Rényi")

    #network = transforma_csv(dir,"Barabási-Albert.csv","Barabási-Albert_transformado.csv")
    #G = estudiar_grafo(dir+"Barabási-Albert_transformado.csv","Barabási-Albert",dir)
    #dibujar_grafo(G,dir,"Barabási-Albert")
    #paradoja_amistad(network,grado_medio(G),"Barabási-Albert")

    network = transforma_csv(dir,"facebook_combined.txt","facebook_combined_transformado.csv"," ")
    G,avg= estudiar_grafo(dir+"facebook_combined_transformado.csv","facebook_combined",dir)
    dibujar_grafo(G,dir,"facebook_combined")
    #paradoja_amistad(network,avg,"facebook_combined")


def transforma_csv(dir,csv,nombre,delim=","):
    network = UndirectedSocialNetwork(dir+csv, delimiter=delim, parse=0)
    visitados = set()
    f = open(dir+nombre,'w')
    for u in network.users():
        f.write(u)
        for v in network.contacts(u):
            if v in visitados:
                continue
            f.write(" ")
            f.write(v)
        f.write('\n')
        visitados.add(u)
    f.close()
    return network

def estudiar_grafo(csv,nombre,dir):
    print("Estudiando grafo " + nombre)
    start = time.process_time()
    G = nx.read_adjlist(csv,",")
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    fig = plt.figure("Degree of "+ nombre, figsize=(8, 8))
    avg = grado_medio(G)
    fig.suptitle("Average Degree: " + str(avg))
    ax1 = fig.add_subplot()
    ax1.bar(*np.unique(degree_sequence, return_counts=True))
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    ax1.set_yscale("log")
    plt.savefig(dir+nombre+"degree.png")

    plt.close()

    timer(start)

    return G,avg

def dibujar_grafo(grafo,dir,nombre):
    print("Dibujando grafo " + nombre)
    start = time.process_time()
    pos = nx.spring_layout(grafo)
    if nombre == "facebook_combined":
        nx.draw_networkx_nodes(grafo,pos,node_size=10)
        nx.draw_networkx_edges(grafo,pos,width=0.2)
    else:
        nx.draw_networkx_nodes(grafo,pos,node_size=100)
        nx.draw_networkx_edges(grafo,pos,width=0.5)

    plt.savefig(dir+nombre+".png")

    timer(start)

def grado_medio(G):
    print("Calculando grado medio")
    start = time.process_time()
    sum = 0
    for u in nx.nodes(G):
        sum+=G.degree[u]

    timer(start)
    return sum/nx.number_of_nodes(G)


def paradoja_amistad(network,avg,nombre):
    print("Paradoja de la amistad para " +nombre)
    start = time.process_time()
    mu1 = {}
    degrees = []

    for u in network.users():
        sum=0
        degrees.append(network.degree(u))
        for v in network.contacts(u):
            sum+=network.degree(v)
        mu1[u] = sum/network.degree(u)

    mu = 0
    for u in mu1:
        mu += mu1[u]

    mu = mu/len(network.users())
    median = sc.median(degrees)
    if avg <= mu:
        print("Se cumple la primera afirmación:"+ str(avg)+" <= " + str(mu))
    if avg >= median:
        print("Se cumple la segunda afirmación:"+ str(avg)+" >= " + str(median))

    timer(start)


def timer(start):
    print("--> elapsed time:", datetime.timedelta(seconds=round(time.process_time() - start)), "<--")

main()
