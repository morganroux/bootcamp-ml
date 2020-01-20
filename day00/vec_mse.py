import numpy as np

def vec_mse(y, y_hat):
	return np.dot(np.transpose(y - y_hat), y - y_hat) / len(y)
