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
import datetime
import itertools
import time
import copy

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
        train = copy.deepcopy(self)
        test = copy.deepcopy(self)
        contador = math.floor((1-ratio)*self.tam)
        for i in range(contador):
            user = random.choice(list(train.ratings.keys()))
            item = random.choice(list(train.ratings[user].keys()))
            rating = train.ratings[user][item]

            if user not in test.ratings:
                test.ratings[user] = {}
            test.ratings[user][item] = rating

            train.ratings[user].pop(item)
            if len(train.ratings[user]) == 0:
                train.ratings.pop(user)
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
            if item not in self.training.ratings[v] or score == 0:
                continue
            sc += self.training.ratings[v][item]*score
            den += score
            count += 1
        if count < self.min:
            return 0
        return sc/den

class ItemKNNRecommender(Recommender):
        def __init__(self, ratings, sim):
            super().__init__(ratings)
            self.sim = sim
            self.vecindario = {}

            similitud = self.sim.similitudes()

            for i in similitud:
                self.vecindario[i] = Ranking(len(self.training.items()))
                for j in similitud[i]:
                    self.vecindario[i].add(j,similitud[i][j])

        def score(self,user,item):
            sc = 0
            for j,score in self.vecindario[item]:
                if j not in self.training.ratings[user]:
                    continue
                sc+=self.training.ratings[user][j]*score
            return sc



class CosineUserSimilarity:
    def __init__(self,ratings):

        self.training = ratings.ratings
        self.sim = {}
        self.modulos = {}

        users = list(ratings.users())

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

class PearsonUserSimilarity:
    def __init__(self,ratings):

        self.training = ratings.ratings
        self.sim = {}

        users = list(ratings.users())

        for u in users:
            if u not in self.sim:
                self.sim[u] = {}
            for v in users:
                if u == v:
                    continue
                if v in self.sim[u]:
                    continue
                if v not in self.sim:
                    self.sim[v] = {}
                pear = 0
                mod_u = 0
                mod_v = 0
                prom_u = self.promedio(u)
                prom_v = self.promedio(v)
                for item in self.training[u]:
                    if item in self.training[v]:
                        mod_u += (self.training[u][item] - prom_u)* (self.training[u][item] - prom_u)
                        mod_v += (self.training[v][item] - prom_v)* (self.training[v][item] - prom_v)
                        pear+= (self.training[u][item] - prom_u)*(self.training[v][item] - prom_v)
                if (math.sqrt(mod_u)*math.sqrt(mod_v)) == 0:
                    self.sim[u][v] = 0
                    self.sim[v][u] = 0
                    continue
                self.sim[u][v] = pear/(math.sqrt(mod_u)*math.sqrt(mod_v))
                self.sim[v][u] = pear/(math.sqrt(mod_u)*math.sqrt(mod_v))


    def promedio(self,user):
        pr = 0
        for item in self.training[user]:
            pr+=self.training[user][item]

        return pr/len(self.training[user])

    def similitudes(self):
        return self.sim


class CosineItemSimilarity:
    def __init__(self,ratings):

        self.training = ratings.ratings
        self.sim = {}
        self.modulos = {}

        items = ratings.items()

        for i in items:
            if i not in self.sim:
                self.sim[i] = {}
            if i not in self.modulos:
                self.modulos[i] = self.module(self.training,i)
            for j in items:
                if i == j:
                    continue
                if j in self.sim[i]:
                    continue
                if j not in self.sim:
                    self.sim[j] = {}
                cos = 0
                if j not in self.modulos:
                    self.modulos[j] = self.module(self.training,j)
                for user in self.training:
                    if i in self.training[user] and j in self.training[user]:
                        cos+= self.training[user][i]*self.training[user][j]
                self.sim[i][j] = cos/(self.modulos[i]*self.modulos[j])
                self.sim[j][i] = cos/(self.modulos[i]*self.modulos[j])


    def module(self,dic,i):
        res = 0
        for user in dic:
            if i in dic[user]:
                res+=dic[user][i]*dic[user][i]
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
        self.test = test.ratings
        self.cutoff = cutoff

    def __repr__(self):
        return type(self).__name__ + ("@" + str(self.cutoff) if self.cutoff != math.inf else "")

    # Esta función se puede dejar abstracta declarándola @abstractmethod,
    # pero también se puede meter algo de código aquí y el resto en las
    # subclases - a criterio del estudiante.
    @abstractmethod
    def compute(self, recommendation):
        return

class Precision(Metric):
    def __init__(self,test, cutoff, threshold=0):
        self.threshold = threshold
        super().__init__(test,cutoff)

    def compute(self, recommendation):
        pr = 0
        rel = 0
        pos = 1
        for user in self.test:
            print(user,recommendation[user])
            for score,item in recommendation[user]:
                if item in self.test[user]:
                    if self.test[user][item] >= self.threshold:
                        print(user,score,self.test[user][item] )
                        rel+=1
                if pos == self.cutoff:
                    break
                else:
                    pos+=1

            pr+=rel/self.cutoff
        return pr/len(self.test.keys())


class Recall(Metric):
    def __init__(self,test, cutoff, threshold=0):
        self.threshold = threshold
        self.relevantes = {}
        super().__init__(test,cutoff)
        for user in self.test:
            self.relevantes[user] = 0
            for item in self.test[user]:
                if self.test[user][item] >= self.threshold:
                    self.relevantes[user] += 1

    def compute(self, recommendation):
        pr = 0
        rel = 0
        pos = 1
        for user in self.test:
            for item,score in recommendation[user]:
                if item in self.test[user]:
                    if self.test[user][item] >= self.threshold:
                        rel+=1
                if pos == self.cutoff:
                    break
                else:
                    pos+=1
            if self.relevantes[user] != 0:
                pr+= (rel/self.relevantes[user])

        return pr/len(self.test.keys())



class RMSE(Metric):
    def __init__(self,test):
        super().__init__(test,0)

    def compute(self, recommendation):
        suma = 0
        for user in self.test:
            for item in self.test[user]:
                if item in recommendation[user]:
                    suma+=(self.test[user][item]-recommendation[user][item])*(self.test[user][item]-recommendation[user][item])
                else:
                    suma+=self.test[user][item]*self.test[user][item]
        return math.sqrt(suma/len(self.test.keys()))



# ItemNNRecomender
# Pearson
# RMSE
def student_test():
    print("=========================\nTesting toy dataset")
    test_dataset("data/toy-ratings.dat", 1, 2, k=4, min=2, topn=4, cutoff=4)
    print("=========================\nTesting MovieLens \"latest-small\" dataset")
    test_dataset("data/ratings.csv", 35, 1240, k=10, min=3, topn=5, cutoff=5, delimiter=',')


def test_dataset(ratings_file, user, item, k, min, topn, cutoff, delimiter='\t'):
    ratings = Ratings(ratings_file, delimiter)
    test_recommender(ratings, k, topn)
    # Now produce a rating split to re-run the recommenders on the training data and evaluate them with the test data
    train, test = ratings.random_split(0.8)
    metric = RMSE(test)
    evaluate_recommenders(train, metric, k, min, 2 * topn)  # Double top n to test a slightly deeper ranking



def test_recommender(ratings, k, topn):
    start = time.process_time()
    print("Creating item cosine similarity")
    sim = CosineItemSimilarity(ratings)
    timer(start)

    start = time.process_time()
    print("Creating kNN Item recommender")
    knn = ItemKNNRecommender(ratings, sim)
    timer(start)

    start = time.process_time()
    print("Testing", knn, "(top", str(topn) + ")")
    recommendation = knn.recommend(topn)
    for user in itertools.islice(recommendation, 4):
        print("    User", user, "->", recommendation[user])
    timer(start)

    start = time.process_time()
    print("Creating user Pearson similarity")
    sim = PearsonUserSimilarity(ratings)
    timer(start)

    start = time.process_time()
    print("Creating kNN User recommender")
    knn = UserKNNRecommender(ratings, sim, k)
    timer(start)

    start = time.process_time()
    print("Testing", knn, "(top", str(topn) + ")")
    recommendation = knn.recommend(topn)
    for user in itertools.islice(recommendation, 4):
        print("    User", user, "->", recommendation[user])
    timer(start)

def evaluate_recommenders(training, metric, k, min, topn):
    print("-------------------------")
    start = time.process_time()
    evaluate_recommender(RandomRecommender(training), topn, metric)
    evaluate_recommender(MajorityRecommender(training, threshold=4), topn, metric)
    evaluate_recommender(AverageRecommender(training, min), topn, metric)
    sim = CosineUserSimilarity(training)
    knn = UserKNNRecommender(training, sim,k)
    evaluate_recommender(knn, topn, metric)
    evaluate_recommender(NormUserKNNRecommender(training, sim, k, min), topn, metric)



def evaluate_recommender(recommender, topn, metric):
    print("Evaluating", recommender)
    recommendation = recommender.recommend(topn)

    print("   ", metric, "=", metric.compute(recommendation))


def timer(start):
    print("--> elapsed time:", datetime.timedelta(seconds=round(time.process_time() - start)), "<--")
