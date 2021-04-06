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
    def __init__(self, file="", delim='\t'):
        self.ratings = {}
        self.tam = 0
        file = open(p, 'r')
        lines = file.readlines()
        for line in lines:
            # info[0] es la persona, info[1] es el elemento a evaluar e info[2]
            # es el rating
            info = line.split(delim)
            self.rate(item[0],item[1],item[2])
        file.close()

    def rate(self, user, item, rating):
        if item not in self.ratings:
            self.ratings[item] = {}
        self.ratings[item][user] = rating
        self.tam += 1

    def rating(self, user, item):
        return self.ratings[item][user]

    def random_split(self, ratio):
        train = self.ratings.copy()
        test = {}
        contador = math.floor((1-ratio)*self.tam)
        for i in range(contador):
            item = random.choice(list(train.keys()))
            user = random.choice(list(train[item].keys()))
            rating = train[items][user]

            if item not in test:
                test[item] = {}
            test[item][user] = rating

            train[item].pop(user)
            if len(train[item]) == 0:
                train.pop(item)
         return train, test

class Ranking:
    def __init__(self, topn):
        self.heap = []
        self.topn = topn

    def add(self, item, score):
        if (len(self.heap) < self.topn) or (score > self.heap[0][0]):
            if len(self.heap) == self.topn:
                heapq.heappop(self.heap)
            heapq.heappush(self.heap, (score, item))

    def __iter__(self):
        h = self.heap.copy()
        ranking = []
        while h:
            ranking.append(heapq.heappop(h)[::-1])
        return reversed(ranking)

    def __repr__(self):
        r = "<"
        for item, score in self:
            r = r + str(item) + ":" + str(score) + " "
        return r[0:-1] + ">"


class Recommender(ABC):
    def __init__(self, training):
        self.training = training

    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def score(self, user, item):
        """ Core scoring function of the recommendation algorithm """

    def recommend(self, topn):
        """ Completar """


class RandomRecommender(Recommender):
    def score(self, user, item):
        return random.random()


class MajorityRecommender(Recommender):
    def __init__(self, ratings, threshold=0):
        super().__init__(ratings)
        self.threshold = threshold

    def score(self, user, item):
        return len(list(rating > self.threshold for user, rating in self.training.item_users(item).items()))


class UserSimilarity(ABC):
    @abstractmethod
    def sim(self, user1, user2):
        """ Computation of user-user similarity metric """


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
