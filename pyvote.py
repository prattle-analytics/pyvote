
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
        self.num_classes = kwargs.get('num_classes')
        self.class_maps = kwargs.get(
            'class_maps')  # if Y predictions for models are not in the same order, pass a class map to ensure correct voting

    def make_predictions(self):
        self.res_dict = dict()
        for i, model in enumerate(self.models):
            if 'keras' in str(model):
                res = model.predict(self.data[i])
                res_list = list()
                doc_idx = 0
                for x in res:
                    idx = (-x).argsort()[:self.top_classes]
                    if self.class_maps:
                        idx = [self.class_maps[i][x] for x in list(idx)]
                    numies = np.flipud(np.sort(x)[-self.top_classes:])
                    if i == 0:
                        self.res_dict[doc_idx] = list(zip(list(idx), list(numies)))
                    else:
                        self.res_dict[doc_idx].extend(zip(list(idx), list(numies)))
                    doc_idx += 1
            else:
                doc_idx = 0
                res = model.predict_proba(self.data[i])
                for i, doc in enumerate(range(0, self.num_classes)):
                    this_doc = res[i]
                    class_arr = np.asarray(([x[0] for x in this_doc]))
                    idx = prob_arr.argsort()[-top_classes:][::-1]
                    prob_arr = np.asarray(([x[1] for x in this_doc]))[idx]

                    if i == 0:
                        self.res_dict[doc_idx] = list(zip(list(idx), list(numies)))
                    else:
                        self.res_dict[doc_idx].extend(zip(list(idx), list(numies)))
                    doc_idx += 1

    def tally_votes(self):
        self.final_list = list()
        for k, v in self.res_dict.items():
            cnts = Counter([x[0] for x in v])
            cnts = sorted(cnts.items(), key=lambda x: x[1], reverse=True)

            votes = list()
            for cnt in cnts:
                if cnt[1] > 1:
                    votes.append(cnt[0])
                if len(votes) == self.top_classes:
                    break

            if len(votes) < self.top_classes:
                probs = [x for x in v if x[0] not in votes]
                probs = sorted(probs, key=lambda tup: tup[1], reverse=True)[0:self.top_classes - len(votes)]
                votes.extend([x[0] for x in probs])
            self.final_list.append(votes)
        return self.final_list














