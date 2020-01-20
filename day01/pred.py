import numpy as np

def predict_(theta, X):
	n = X.shape[0]
	one = np.array([1 for i in range(n)]).reshape(n,1)
	X1 = np.hstack((one, X))
	return X1.dot(theta)

if __name__ == "__main__":
	X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
	theta3 = np.array([[0.05], [1.], [1.], [1.]])
	print(predict_(theta3, X3))
