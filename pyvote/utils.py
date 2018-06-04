import numpy as np

def rowwise_indexed(mat, idx):
	ran = np.arange(mat.shape[0]).reshape(-1, 1)
	ran = np.repeat(ran, mat.shape[1], axis=1)
	return mat[ran, idx]

def columnwise_indexed(mat, idx):
	ran = np.arange(mat.shape[1]).reshape(1, -1)
	ran = np.repeat(ran, mat.shape[0])
	return mat[ran, idx]