
from operator import itemgetter
from collections import Counter
import numpy as np 

class ModelVote(object):

    '''
    suppported model types: keras, sklearn
    '''


    def __init__(self, **kwargs):

        self.models = kwargs.get('models')
        self.data = kwargs.get('data')
        self.class_type = kwargs.get('class_type', 'binary')
        self.top_classes = kwargs.get('top_classes', 5)
        self.class_maps = kwargs.get \
            ('class_maps')  # if Y predictions for models are not in the same order, pass a class map to ensure correct voting


    def make_predictions(self):
        res_mat = None
        for i, model in enumerate(self.models):
            res = model.predict(self.data[i])
            cats = (-res.argsort()).argsort()
            probs = np.fliplr(np.sort(res, axis=1))
            cats, probs = map(lambda x: x[:, self.top_classes], (cats, probs))
            cats, probs = map(lambda x: x.reshape(x.shape[0], 1, -1), (cats, probs))
            if self.class_maps:
                for ind, clas in enumerate(self.class_maps[i]):
                    cats[cats == ind] = clas

            votes = np.concatenate((cats, probs), axis=1)
            votes = np.transpose(votes, (0, 2, 1))


            if res_mat is None:
                res_mat = votes
            else:
                res_mat = np.concatenate((res_mat, votes), axis=1)

        self.res_dict = dict(enumerate(res_mat))


    def tally_votes(self):
        self.final_list = list()
        for k, v in self.res_dict.items():
            cnts = Counter([x[0] for x in v])
            cnts = sorted(cnts.items(), key=lambda x :x[1], reverse=True)

            votes = list()
            for cnt in cnts:
                if cnt[1] > 1:
                    votes.append(cnt[0])
                if len(votes) == self.top_classes:
                    break

            if len(votes) < self.top_classes:
                probs = [x for x in v if x[0] not in votes]
                probs = sorted(probs, key=lambda tup: tup[1], reverse=True)[0:self. top_classes -len(votes)]
                votes.extend([x[0] for x in probs])
            self.final_list.append(votes)
        return self.final_list







