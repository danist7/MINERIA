import heapq
from abc import ABC, abstractmethod
import math
import random

class UndirectedSocialNetwork:
    def __init__(self, file, delimiter='\t', parse=0):
        self.sn = {}
        self.edges = 0
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            u1,u2 = line.split(delimiter)
            if parse != 0:
                u1 = parse(u1)
                u2 = parse(u2)
            else:
                u1 = u1.rstrip()
                u2 = u2.rstrip()
            if u1 not in self.sn:
                self.sn[u1] = set()
            if u2 not in self.sn:
                self.sn[u2] = set()
            if u1 in self.sn[u2] and u2 in self.sn[u1]:
                continue
            self.sn[u1].add(u2)
            self.sn[u2].add(u1)
            self.edges+=1
        f.close()

    def users(self):
        return list(self.sn.keys())

    def contacts(self, user):
        return self.sn[user]

    def degree(self, user):
        return len(self.sn[user])

    def add_contact(self, u, v):
        if u not in self.sn:
            self.sn[u] = set()
        if v not in self.sn:
            self.sn[v] = set()
        self.sn[u].add(v)
        self.sn[v].add(u)
        self.edges += 1

    def connected(self, u, v):
        if u in self.sn[v] or v in self.sn[u]:
            return True
        else:
            return False

    def nedges(self):
        return self.edges


class Metric(ABC):
    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def compute_all(self, network):
        """" Compute metric on all users or edges of network """


class LocalMetric(Metric):
    def __init__(self, topn=5):
        self.topn = topn

    @abstractmethod
    def compute(self, network, element):
        """" Compute metric on one user of edge of network """

class UserClusteringCoefficient(LocalMetric):

    def compute(self, network, element):
        grado = network.degree(element)
        den = (grado*(grado-1))/2
        if den == 0:
            return 0
        num = 0
        vecinos = list(network.contacts(element))
        for i in range(len(vecinos)):
            for j in range(i+1,len(vecinos)):
                if vecinos[j] in network.contacts(vecinos[i]):
                    num += 1
        return num/den

    def compute_all(self,network):
        rank = Ranking(self.topn)
        for u in network.users():
            score = self.compute(network,u)
            rank.add(u,score)
        return rank

class ClusteringCoefficient(Metric):

    def compute_all(self,network):
        triangulos = []
        tripletas = []
        for u1 in network.users():
            for u2 in network.contacts(u1):
                for u3 in (network.contacts(u2) - set((u1))):
                    if (u1,u2,u3) in tripletas:
                        continue
                    tripletas.append((u1,u2,u3))
                    if u1 in network.contacts(u3):
                        if set((u1,u2,u3)) in triangulos:
                            continue
                        triangulos.append(set((u1,u2,u3)))
                        tripletas.append((u1,u3,u2))
                        tripletas.append((u2,u3,u1))
                        tripletas.append((u2,u1,u3))
                        tripletas.append((u3,u1,u2))
                        tripletas.append((u3,u2,u1))


        return (2*3*len(triangulos))/len(tripletas)

class Embeddedness(LocalMetric):
    # Element es el par (u,v) del que calcular la métrica
    def compute(self, network, element):
        union = (network.contacts(element[0]) - set([element[1]])) | (network.contacts(element[1]) - set([element[0]]))
        inter = (network.contacts(element[0]) - set([element[1]])) & (network.contacts(element[1]) - set([element[0]]))
        if len(union) == 0:
            return 0
        return len(inter)/len(union)

    def compute_all(self,network):
        rank = Ranking(self.topn)
        recorridos = set()
        for u in network.users():
            for v in network.users():
                if v in recorridos or v == u:
                    continue
                score = self.compute(network,(u,v))
                rank.add((u,v),score)
            recorridos.add(u)
        return rank

class Assortativity(Metric):
    def compute_all(self,network):
        m = network.nedges()
        sum1 = 0
        sum2 = 0
        sum3 = 0
        recorridos = set()
        for u in network.users():
            for v in network.contacts(u):
                if v in recorridos:
                    continue
                sum1 += network.degree(u)*network.degree(v)
            recorridos.add(u)
            sum2 += math.pow(network.degree(u),2)
            sum3 += math.pow(network.degree(u),3)
        return (4*m*sum1 - math.pow(sum2,2))/(2*m*sum3 - math.pow(sum2,2))


class AvgUserMetric(Metric):
    def __init__(self,metric):
        self.metric = metric

    def compute_all(self,network):
        res = 0
        for user in network.users():
            res += self.metric.compute(network,user)
        return res/len(network.users())
class Closeness(LocalMetric):
    def compute(self, network, element):
        dij = Dijkstra()
        dist = dij.compute(network,element)
        suma = 0
        for u in dist:
            suma += dist[u]
        return (len(network.users()) - 1)/suma

        def compute_all(self,network):
            rank = Ranking(self.topn)
            for u in network.users():
                score = self.compute(network,u)
                rank.add(u,score)

            return rank
class AverageShortestPath(Metric):
    def compute_all(self,network):
        suma = 0
        recorridos = set()
        dij = Dijkstra()
        n = len(network.users())
        for u in network.users():
            dist = dij.compute(network,u)
            for v in dist:
                if v in recorridos:
                    continue
                suma += dist[v]
            recorridos.add(u)
        return suma*(2/(n*(n-1)))
class Dijkstra:
    def __init__(self):
        return
    def compute(self,network, user):
        distancias = {}
        marcados = set()
        distancias[user] = 0
        a = user

        while len(marcados) < network.users():
            for u in network.contacts(user):
                if u in marcados:
                    continue
                if u not in distancias:
                    distancias[u] = distancias[a]+1
                    continue
                dist = distancias[a] + 1
                if distancias[u] > dist:
                    distancias[u] = dist
            marcados.add(a)
            a = sorted(distancias.items(), key=lambda item: item[1])[0][0]
        return distancias

class Amigos_Comunes:
    def __init__(self,num_nodos,p,q):
        self.num_nodos = num_nodos
        self.p = p
        self.q = q
        self.sn = {}
        self.edges = 0
        for i in range(self.num_nodos):
            self.sn[i] = set()

    def create(self):
        for i in range(self.num_nodos):
            if i == 0:
                for j in range(1,self.num_nodos):
                    if p <= random.random():
                        self.sn[i].add(j)
                        self.sn[j].add(i)
                        edges += 1
            else:
                for j in range(self.num_nodos):
                    vecino = 0
                    continuar = 0
                    if i == j:
                        continue
                    for nodo in self.sn[i]:
                        if j in self.sn[nodo]:
                            vecino = 1
                            break
                    if vecino:
                        if q<= random.random():
                            if p <= random.random():
                                self.sn[i].add(j)
                                self.sn[j].add(i)
                                edges += 1
                    else:
                        if q>random.random():
                            if p <= random.random():
                                self.sn[i].add(j)
                                self.sn[j].add(i)
                                edges += 1
        return

    def to_csv(self,file,delimiter='\t'):
        f = open(file,'w')
        for user1 in self.sn:
            for user2 in self.sn[user1]:
                if user2 < user1:
                    continue
                linea = str(user1) + delimiter + str(user2) + "\n"
                f.write(linea)
        f.close()

class Ranking:
    class ScoredUser:
        """
        Clase utilizada para gestionar las comparaciones que se realizan dentro del heap
        """
        def __init__(self, element):
            self.element = element

        def __lt__(self, other):
            """
            En primer lugar se compara el score. En caso de que sean iguales (mismo score),
            se compara usando el userid (se colocará más arriba el elemento con un userid menor).
            """
            return self.element[0] < other.element[0] if self.element[0] != other.element[0] \
                else self.element[1] > other.element[1]

        def __eq__(self, other):
            return self.element == other.element

        def __str__(self):
            return str(self.element)

        def __repr__(self):
            return self.__str__()

    def __init__(self, topn):
        self.heap = []
        self.topn = topn
        self.changed = 0

    def add(self, user, score):
        scored_user = self.ScoredUser((score, user))
        if len(self.heap) < self.topn:
            heapq.heappush(self.heap, scored_user)
            self.changed = 1
        elif scored_user > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, scored_user)
            self.changed = 1

    def __iter__(self):
        if self.changed:
            self.ranking = []
            h = self.heap.copy()
            while h:
                self.ranking.append(heapq.heappop(h).element[::-1])
            self.changed = 0
        return reversed(self.ranking)

    def __repr__(self):
        r = "<"
        for user, score in self:
            r += str(user) + ":" + str(score) + " "
        return r[0:-1] + ">"

# Closeness
# Average Shortest Path
# Amigos_comunes
def student_test():
    test_network("graph/small1.csv", ",", 5, 6, 4, int)
    test_network("graph/small2.csv", ",", 5, 3, 5, int)
    test_network("graph/small3.csv", ",", 5, "a", "b")

    print("Creating amigos comunes")
    start = time.process_time()
    amigos = Amigos_Comunes(50,0.7,0.9)
    amigos.create()
    amigos.to_csv("amigos_comunes.csv")
    timer(start)

def test_network(file, delimiter, topn, u, v, parse=0):
    print("==================================================\nTesting " + file + " network")
    network = UndirectedSocialNetwork(file, delimiter=delimiter, parse=parse)
    print(len(network.users()), "users and", network.nedges(), "contact relationships")
    print("User", u, "has", network.degree(u), "contacts")

    print("-------------------------")
    test_metric(Closeness(topn), network, u)

    print("-------------------------")
    test_global_metric(AverageShortestPath(), network)



def test_metric(metric, network, example):
    start = time.process_time()
    print(metric, ":", metric.compute_all(network))
    print(str(metric) + "(" + str(example) + ") =", metric.compute(network, example))
    timer(start)

def test_global_metric(metric, network):
    start = time.process_time()
    print(metric, "=", metric.compute_all(network))
    timer(start)

def timer(start):
    print("--> elapsed time:", datetime.timedelta(seconds=round(time.process_time() - start)), "<--")
