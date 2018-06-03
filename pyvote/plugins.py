
def get_plugin(model):
	try:
		from sklearn.base import BaseEstimator
	except ImportError:
		pass
	else:
		if isinstance(model, BaseEstimator):
			return SKLearnPlugin(model)

	return BasePlugin(model)


class BasePlugin(object):

	def __init__(self, model):
		self.model = model

	def predict(self, data):
		return self.model.predict(data)

	def predict_clean(self, data):
		return self.predict(data)

	def get_classes(self):
		raise NotImplementedError

	def sub_cats(self, cats):
		try:
			classes = self.get_classes()
			assert classes is not None
		except (NotImplementedError, AssertionError):
			return cats
		
		for ind, cat in enumerate(classes):
			cats[cats == ind] = cat
		return cats

class SKLearnPlugin(BasePlugin):

	def get_classes(self):
		if hasattr(self.model, 'n_classes_'):
			n_classes = self.model.n_classes_
			if isinstance(n_classes, int):
				return self.model.classes_

		else:
			return self.model.classes_

	def predict(self, data):
		probs = self.model.predict_proba(data)
		if isinstance(probs, list):
			probs = np.array(probs)[:, :, 0].T
		return probs