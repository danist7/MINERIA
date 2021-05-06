import networkx as nx
g = nx.barabasi_albert_graph(50,5)


nx.write_adjlist(g,"adj.csv")
