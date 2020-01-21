import numpy as np

def vec_gradient(X, Y, theta):
	m = X.shape[0]
	n= X.shape[1]
	return 1/m * np.transpose(X).dot(X.dot(theta) - Y)

if __name__ == "__main__":
	X = np.array([
    [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	Z = np.array([3,0.5,-6])
	print(vec_gradient(X, Y, Z))
