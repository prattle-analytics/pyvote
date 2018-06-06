import numpy as np

def rowwise_indexed(mat, idx):
	ran = np.arange(idx.shape[0]).reshape(-1, 1)
	ran = np.repeat(ran, idx.shape[1], axis=1)
	return mat[ran, idx]

def columnwise_indexed(mat, idx):
	return rowwise_indexed(mat.T, idx.T).T

def sub_cats(mat, cats, reverse=False):
	mat_bak = mat.copy()
	for ind, cat in enumerate(cats):
		if reverse:
			mat[mat_bak == cat] = ind
		else:
			mat[mat_bak == ind] = cat
	del mat_bak
	return mat