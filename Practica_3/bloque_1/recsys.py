"""
 Copyright (C) 2021 Pablo Castells y Alejandro Bellogín

 Este código se ha implementado para la realización de las prácticas de
 la asignatura "Búsqueda y minería de información" de 4º del Grado en
 Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 se ciñe a las actividades docentes de dicha asignatura.
"""

import heapq
from abc import ABC, abstractmethod
import math
import random

class Ratings:
    def __init__(self, p="", delim='\t'):
        self.ratings = {}
        self.tam = 0
        self.items_names = set()
        file = open(p, 'r')
        lines = file.readlines()
        for line in lines:
            # info[0] es la persona, info[1] es el elemento a evaluar e info[2]
            # es el rating
            info = line.split(delim)
            self.rate(int(info[0]),int(info[1]),float(info[2]))
            self.items_names.add(int(info[1]))
        file.close()

    def rate(self, user, item, rating):
        if user not in self.ratings:
            self.ratings[user] = {}
        self.ratings[user][item] = rating
        self.tam += 1

    def rating(self, user, item):
        return self.ratings[user][item]

    def nratings(self):
        nrating = 0
        for user in self.ratings:
            for item in self.ratings[user]:
                nrating += 1
        return nrating

    def users(self):
        return list(self.ratings.keys())

    def items(self):
        return list(self.items_names)

    def user_items(self,user):
        if user not in self.ratings:
            return
        return self.ratings[user]

    def item_users(self,item):
        ret = {}
        for u in self.ratings:
            if item in self.ratings[u]:
                ret[u] = self.ratings[u][item]
        return ret

    def random_split(self, ratio):
        train = self.ratings.copy()
        test = {}
        contador = math.floor((1-ratio)*self.tam)
        for i in range(contador):
            user = random.choice(list(train.keys()))
            item = random.choice(list(train[user].keys()))
            rating = train[user][item]

            if user not in test:
                test[user] = {}
            test[user][item] = rating

            train[user].pop(item)
            if len(train[user]) == 0:
                train.pop(user)
        return train, test



class Ranking:
    class ScoredItem:
        """
        Clase utilizada para gestionar las comparaciones que se realizan dentro del heap
        """
        def __init__(self, element):
            self.element = element

        def __lt__(self, other):
            """
            En primer lugar se compara el score. En caso de que sean iguales (mismo score),
            se compara usando el itemid (se colocará más arriba el elemento con un itemid menor).
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

    def add(self, item, score):
        scored_item = self.ScoredItem((score, item))
        if len(self.heap) < self.topn:
            heapq.heappush(self.heap, scored_item)
            self.changed = 1
        elif scored_item > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, scored_item)
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
        for item, score in self:
            r = r + str(item) + ":" + str(score) + " "
        return r[0:-1] + ">"


class Recommender(ABC):
    def __init__(self, training):
        self.training = training
        self.rankings = {}

    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def score(self, user, item):
        """ Core scoring function of the recommendation algorithm """

    def recommend(self, topn):
        ratings = self.training.ratings
        users = ratings.keys()
        items = set(self.training.items())
        for user in users:
            ranked = set(ratings[user].keys())
            to_rank = items.difference(ranked)
            self.rankings[user] = Ranking(topn)
            for item in to_rank:
                self.rankings[user].add(item,self.score(user,item))
        return self.rankings

class RandomRecommender(Recommender):
    def score(self, user, item):
        return random.random()


class MajorityRecommender(Recommender):
    def __init__(self, ratings, threshold=0):
        super().__init__(ratings)
        self.threshold = threshold

    def score(self, user, item):
        return sum(rating >= self.threshold for user, rating in self.training.item_users(item).items())

class AverageRecommender(Recommender):
    def __init__(self, ratings, threshold=0):
        super().__init__(ratings)
        self.threshold = threshold

    def score(self, user, item):
        cont = 0
        sum = 0
        for u in self.training.ratings:
            if item in self.training.ratings[u]:
                sum += self.training.ratings[u][item]
                cont += 1
        if cont < self.threshold:
            return 0
        else:
            return sum/cont


class UserKNNRecommender(Recommender):
    def __init__(self, ratings, sim, k):
        super().__init__(ratings)
        self.sim = sim
        self.k = k
        self.vecindario = {}

        similitud = self.sim.similitudes()

        for u in similitud:
            self.vecindario[u] = Ranking(k)
            for v in similitud[u]:
                self.vecindario[u].add(v,similitud[u][v])


    def score(self,user,item):
        sc = 0
        for v,score in self.vecindario[user]:
            if item not in self.training.ratings[v]:
                continue
            sc+=self.training.ratings[v][item]*score
        return sc

class NormUserKNNRecommender(UserKNNRecommender):
    def __init__(self, ratings, sim, k, min):
        self.min = min
        super().__init__(ratings, sim, k)

    def score(self,user,item):
        sc = 0
        den = 0
        count = 0
        for v,score in self.vecindario[user]:
            if item not in self.training.ratings[v]:
                continue
            sc += self.training.ratings[v][item]*score
            den += score
            count += 1
        if count < self.min:
            return 0
        return sc/den

class CosineUserSimilarity:
    def __init__(self,ratings):

        self.training = ratings.ratings
        self.sim = {}
        self.modulos = {}

        users = list(self.training.keys())

        for u in users:
            if u not in self.sim:
                self.sim[u] = {}
            if u not in self.modulos:
                self.modulos[u] = self.module(self.training[u])
            for v in users:
                if u == v:
                    continue
                if v in self.sim[u]:
                    continue
                if v not in self.sim:
                    self.sim[v] = {}
                cos = 0
                if v not in self.modulos:
                    self.modulos[v] = self.module(self.training[v])
                for item in self.training[u]:
                    if item in self.training[v]:
                        cos+= self.training[u][item]*self.training[v][item]
                self.sim[u][v] = cos/(self.modulos[u]*self.modulos[v])
                self.sim[v][u] = cos/(self.modulos[u]*self.modulos[v])

    def module(self,dic):
        res = 0
        for item in dic:
            res+=dic[item]*dic[item]
        return math.sqrt(res)

    def similitudes(self):
        return self.sim

class UserSimilarity(ABC):
    @abstractmethod
    def sim(self, user1, user2):
        """ Computation of user-user similarity metric """
        return


class Metric(ABC):
    def __init__(self, test, cutoff):
        self.test = test
        self.cutoff = cutoff

    def __repr__(self):
        return type(self).__name__ + ("@" + str(self.cutoff) if self.cutoff != math.inf else "")

    # Esta función se puede dejar abstracta declarándola @abstractmethod,
    # pero también se puede meter algo de código aquí y el resto en las
    # subclases - a criterio del estudiante.
    def compute(self, recommendation):
        """ Completar """
        return
