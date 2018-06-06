from operator import itemgetter
from collections import Counter
import numpy as np 
import pandas
from .plugins import get_plugin, BasePlugin
from .utils import rowwise_indexed, sub_cats

class Predictions(object):

    def __init__(self, pred_mat, probs, cats, voter):
        self.pred_mat = pred_mat
        self.probs = probs
        self.voter = voter
        self.cats = cats

    def tally_votes(self, class_type='binary', top_classes=5, min_votes=1, return_means=False):
        pred_mat = self.pred_mat.copy()
        pred_mat = pred_mat[:, :, :top_classes, :]

        shp = pred_mat.shape
        all_votes = pred_mat.reshape(shp[0], -1, shp[-1])

        votes_list = []
        for k, v in enumerate(all_votes):
            cnts = np.unique(v[:, 0], return_counts=True)
            cnts = np.vstack(cnts).T
            cnts = cnts[(-cnts[:, 1]).argsort()]

            votes = cnts[cnts[:, 1] > min_votes][:top_classes, 0]

            if len(votes) < top_classes:
                probs = v[~np.isin(v[:, 0], votes)]
                probs = probs[(-probs[:, 1]).argsort()]

                need = top_classes - len(votes)
                probs = np.unique(probs[:, 0])[:need]
                votes = np.concatenate((votes, probs))

            votes = votes.reshape(1, -1)
            votes_list.append(votes)

        inds_mat = np.concatenate(votes_list).astype(int)
        votes_mat = sub_cats(votes_mat.copy(), self.cats)

        if not return_means:
            return votes_mat

        avg_prob = self.probs.mean(axis=1)
        return votes_mat, rowwise_indexed(avg_prob, inds_mat)


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
        all_probs, all_convs = [], []
        for i, model in enumerate(self.models):
            model_data = self.datagetter[i](data)
            plugin = model
            if not isinstance(plugin, BasePlugin):
                plugin = get_plugin(plugin)

            res = plugin.predict(model_data)

            conv = np.arange(cats.shape[-1])
            conv = plugin.sub_cats(conv)

            conv_bak = conv.copy()
            if self.class_maps and self.class_maps[i]:
                for ind, clas in enumerate(self.class_maps[i]):
                    conv[conv_bak == ind] = clas

            all_convs.append(conv)
            all_probs.append(res)

        cats = set().union(*all_convs)

        full_convs, full_probs = [], []
        for conv, probs in zip(all_convs, all_probs):
            extra_cats = np.array(list(cats - set(conv)))
            conv = np.concatenate((conv, extra_cats))

            extra_probs = np.zeros((probs.shape[0], len(extra_cats)))
            probs = np.concatenate((probs, extra_probs), axis=1)

            full_convs.append(conv)
            full_probs.append(prob)

        del all_convs, all_probs

        cats = np.array(sorted(cats))

        for conv in full_convs:
            sub_cats(conv, cats, reverse=True)

        votes = []
        for conv, probs in zip(full_convs, full_probs):
            cats = (-probs).argsort()
            sprobs = rowwise_indexed(probs, cats)

            cats_bak = cats.copy()
            for ind, cat in enumerate(conv):
                cats[cats_bak == ind] = cat
            del cats_bak

            cats, sprobs = map(lambda x: x.reshape(-1, 1, x.shape[1]), (cats, sprobs))
            v = np.concatenate((cats, sprobs), axis=1)
            v = v.reshape(-1, 1, *v.shape[1:])
            votes.append(v)

        votes = np.concatenate(votes, axis=1)

        idx = votes[..., 0].argsort()
        idx = idx.reshape(-1, idx.shape[-1])
        idx += np.arange(idx.shape[0]).reshape(-1, 1) * idx.shape[1]
        idx = idx.reshape(-1)

        mat = votes.reshape(-1, mat.shape[-1])
        mat = mat[idx]
        mat = mat.reshape(votes.shape)[..., 1]

        return Predictions(votes, mat, cats, self)