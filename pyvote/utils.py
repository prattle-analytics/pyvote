import numpy as np

def rowwise_indexed(mat, idx):
	ran = np.arange(idx.shape[0]).reshape(-1, 1)
	ran = np.repeat(ran, idx.shape[1], axis=1)
	return mat[ran, idx]

def columnwise_indexed(mat, idx):
	ran = np.arange(idx.shape[1]).reshape(1, -1)
	ran = np.repeat(ran, idx.shape[0])
	return mat[ran, idx]