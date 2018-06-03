
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

class SKLearnPlugin(BasePlugin):

	def predict(self, data):
		probs = self.model.predict_proba(data)
		if isinstance(probs, list):
			probs = np.array(probs)[:, :, 0].T
		return probs