from recsys import *

student_test()

"""self.training = ratings.ratings
self.sim = {}
self.modulos = {}

items = ratings.items()

for i in items:
    try:
        fl = self.sim[i]
    except Exception as e:
        self.sim[i] = {}

    try:
        fl = self.modulos[i]
    except Exception as e:
        self.modulos[i] = self.module(self.training,i)

    for j in items:
        if i == j:
            continue
        try:
            fl = self.sim[i]
            continue
        except Exception as e:
            pass

        try:
            fl = self.sim[j]
        except Exception as e:
            self.sim[j] = {}

        cos = 0
        if j not in self.modulos:
            self.modulos[j] = self.module(self.training,j)
        for user in self.training:
            if i in self.training[user] and j in self.training[user]:
                cos+= self.training[user][i]*self.training[user][j]
        self.sim[i][j] = cos/(self.modulos[i]*self.modulos[j])
        self.sim[j][i] = cos/(self.modulos[i]*self.modulos[j])"""
