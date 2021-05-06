import heapq
from abc import ABC, abstractmethod
import math

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
    def __init__(self, topn):
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
        visitados = []
        for first in network.contacts(element):
            for second in network.contacts(first):
                par = set([first,second])
                if par in visitados:
                    continue
                visitados.append(par)
                if element in network.contacts(second):
                    num+=1
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
                for u3 in network.contacts(u2):
                    if u1 == u3:
                        continue
                    if (u1,u2,u3) in tripletas:
                        continue
                    tripletas.append((u1,u2,u3))
                    if u1 in network.contacts(u3):
                        if set((u1,u2,u3)) in triangulos:
                            continue
                        triangulos.append(set((u1,u2,u3)))


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
