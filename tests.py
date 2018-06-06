import numpy as np
from pyvote.utils import rowwise_indexed, columnwise_indexed, sub_cats
import unittest

class PyvoteUtilsTestCase(unittest.TestCase):

	def test_rowwise_indexed(self, rowwise=True):
		mat1 = np.array([
			[1, 2, 3],
			[4, 21, 3],
			[1, 23, 3],
			[3, 2, 5]
		])

		idx = np.array([
			[0, 2, 1],
			[2, 0, 2],
			[1, 0, 2],
			[2, 1, 0]
		])

		assert mat1.shape == idx.shape

		if rowwise:
			res = rowwise_indexed(mat1, idx)
		else:
			res = columnwise_indexed(mat1.T, idx.T).T

		should = np.array([
			[1, 3, 2],
			[3, 4, 3],
			[23, 1, 3],
			[5, 2, 3]
		])

		self.assertEqual(res.shape, should.shape, 'The shape was incorrect')
		eq = res == should
		self.assertTrue(res.all(), 'The result matrix was incorrect')

	def test_columnwize_indexed(self):
		self.test_rowwise_indexed(False)

	def test_sub_cats(self, reverse=False):
		cats = np.array([5, 8, 2, 1])

		mat = np.array([
			[3, 1, 3, 2],
			[0, 1, 3, 0],
			[0, 2, 3, 1]
		])

		should = np.array([
			[1, 8, 1, 2],
			[5, 8, 1, 5],
			[5, 2, 1, 8]
		])

		pre, post = mat, should
		kws = {}
		if reverse:
			post, pre = pre, post
			kws = {'reverse': True}

		assert cats.shape[-1] == pre.shape[-1]

		subbed = sub_cats(pre.copy(), cats, **kws)

		self.assertEqual(post.shape, subbed.shape, 'The shape was incorrect')
		eq = post == subbed
		self.assertTrue(eq.all(), 'The result matrix was incorrect')

	def test_sub_cats_revved(self):
		self.test_sub_cats(True)