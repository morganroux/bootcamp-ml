import numpy as np

def predict_(theta, X):
	n = X.shape[0]
	one = np.array([1 for i in range(n)]).reshape(n,1)
	X1 = np.hstack((one, X))
	return X1.dot(theta)

def cost_elem_(theta, X, Y):
	M = X.shape[0]
	Y_pred = predict_(theta, X)
	return 0.5 / M * np.array([(y_pred - y)**2 for y_pred, y in zip(Y_pred, Y)])

def cost_(theta, X, Y):
	M = X.shape[0]
	Y_pred = predict_(theta, X)
	diff = Y_pred - Y
	return 0.5 / M * np.transpose(diff).dot(diff)

#	sum = 0
#	for y in cost_elem_(theta, X, Y):
#		sum += y
#	return sum

if __name__ == "__main__":
	X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
	theta2 = np.array([[0.05], [1.], [1.], [1.]])
	Y2 = np.array([[19.], [42.], [67.], [93.]])
	print(cost_elem_(theta2, X2, Y2))
	print(cost_(theta2, X2, Y2))
