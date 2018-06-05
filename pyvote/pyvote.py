from operator import itemgetter
from collections import Counter
import numpy as np 
from operator import attrgetter
from .plugins import get_plugin, BasePlugin
from .utils import rowwise_indexed, row_conv_to_idx

class Predictions(object):

    def __init__(self, pred_mat, probs, conv_mat, voter):
        self.pred_mat = pred_mat
        self.probs = probs
        self.voter = voter
        self.conv_mat = conv_mat

    def tally_votes(self, class_type='binary', top_classes=5, min_votes=1, return_means=False):
        pred_mat = self.pred_mat
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

        votes_mat = np.concatenate(votes_list).astype(int)

        if not return_means:
            return votes_mat

        avg_prob = self.probs.mean(axis=1)
        ind_mat = row_conv_to_idx(votes_mat, self.conv_mat)
        return votes_mat, rowwise_indexed(avg_prob, ind_mat)


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
        all_probs = []
        all_convs = []
        for i, model in enumerate(self.models):
            model_data = self.datagetter[i](data)
            plugin = model
            if not isinstance(plugin, BasePlugin):
                plugin = get_plugin(plugin)
            res = plugin.predict(model_data)

            cats = (-res).argsort()
            probs = rowwise_indexed(res, cats)
            cats, probs = map(lambda x: x.reshape(-1, 1, x.shape[1]), (cats, probs))

            conv_vec = np.arange(res.shape[1]).reshape(1, -1)
            conv_vec = plugin.sub_cats(conv_vec)

            if self.class_maps and self.class_maps[i]:
                for ind, clas in enumerate(self.class_maps[i]):
                    conv_vec[conv_vec == ind] = clas

            if ind, val in enumerate(conv_vec):
                cats[cats == ind] = val

            votes = np.concatenate((cats, probs), axis=1)
            votes = np.transpose(votes, (0, 2, 1))
            votes = votes.reshape(-1, 1, *votes.shape[1:])

            res = res.reshape(-1, 1, *res.shape[1:])
            all_votes.append(votes)
            all_probs.append(res)
            all_convs.append(conv_vec)

        res_mat = np.concatenate(all_votes, axis=1)
        conv_mat = np.concatenate(all_convs)
        probs = np.concatenate(all_probs, axis=1)

        return Predictions(res_mat, probs, conv_mat, self)

