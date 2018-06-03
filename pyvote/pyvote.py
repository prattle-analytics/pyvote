from operator import itemgetter
from collections import Counter
import numpy as np 
from .plugins import get_plugin

class Predictions(object):

    def __init__(self, pred_mat, voter):
        self.pred_mat = pred_mat
        self.voter = voter

    def tally_votes(self, class_type='binary', top_classes=5, min_votes=1):
        pred_mat = self.pred_mat
        pred_mat = pred_mat[:, :, :top_classes, :]

        shp = pred_mat.shape
        all_votes = pred_mat.reshape(shp[0], -1, shp[-1])

        votes_list = []
        for k, v in enumerate(all_votes):
            cnts = np.unique(v[:, 0], return_counts=True)
            cnts = np.concatenate(cnts, axis=1)
            cnts = cnts[(-cnts[:, 1]).argsort()]

            votes = cnts[cnts[:, 1] > min_votes][:top_classes, 0]

            if len(votes) < self.top_classes:
                probs = v[~np.isin(v[:, 0], votes)]
                probs = probs[(-probs[:, 1]).argsort()]

                need = top_classes - len(votes)
                probs = np.unique(probs[:, 0])[:need]
                votes = np.concatenate((votes, probs))

            votes = votes.reshape(1, -1)
            votes_list.append(votes)

        return np.concatenate(votes_list)

class ModelVote(object):

    '''
    suppported model types: keras, sklearn
    '''


    def __init__(self, models, class_maps=None, datagetter=None):

        self.models = models
        self.class_maps = class_maps # if Y predictions for models are not in the same order, pass a class map to ensure correct voting
        if datagetter is None:
            datagetter = list(map(itemgetter, range(len(models))))
        self.datagetter = datagetter 

    def make_predictions(self, data):
        # matrix shape (n_samples, n_models, n_classes, 2)
        all_votes = []
        for i, model in enumerate(self.models):
            model_data = self.datagetter[i](data)
            plugin = get_plugin(model)
            res = plugin.predict(model_data)

            cats = (-res.argsort()).argsort()
            probs = np.fliplr(np.sort(res, axis=1))
            cats, probs = map(lambda x: x.reshape(-1, 1, x.shape[1]), (cats, probs))

            if self.class_maps and self.class_maps[i]:
                for ind, clas in enumerate(self.class_maps[i]):
                    cats[cats == ind] = clas

            votes = np.concatenate((cats, probs), axis=1)
            votes = np.transpose(votes, (0, 2, 1))
            votes = votes.reshape(-1, 1, *votes.shape[1:])

            all_votes.append(votes)

        res_mat = np.concatenate(all_votes, axis=1)

        return Predictions(res_mat, self)