
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

class SKLearnPlugin(BasePlugin):

	def predict(self, data):
		return 