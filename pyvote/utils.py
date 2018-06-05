import numpy as np
from operator import attrgetter

def iterate_columns(mat):
	return map(attrgetter('T'), mat.T)

def rowwise_indexed(mat, idx):
	ran = np.arange(idx.shape[0]).reshape(-1, 1)
	ran = np.repeat(ran, idx.shape[1], axis=1)
	return mat[ran, idx]

def columnwise_indexed(mat, idx):
	ran = np.arange(idx.shape[1]).reshape(1, -1)
	ran = np.repeat(ran, idx.shape[0])
	return mat[ran, idx]

def row_conv_to_idx(mat, conv):
	ind_mat = np.zeros(mat.shape)
	for ind, col in enumerate(iterate_columns(mat)):
		ind_mat[mat == col] = ind
	return ind_mat

def row_idx_to_conv(mat, conv):
	conv_mat = np.zeros(mat.shape)
	for ind, col in enumerate(iterate_columns(mat)):
		conv_mat[mat == ind] = col
	return conv_mat

def col_conv_to_idx(mat, conv):
	ind_mat = np.zeros(mat.shape)
	for ind, row in enumerate(mat):
		ind_mat[mat == row] = ind
	return ind_mat

def col_idx_to_conv(mat, conv):
	conv_mat = np.zeros(mat.shape)
	for ind, row in enumerate(mat):
		conv_mat[mat == ind] = row
	return conv_mat