import numpy as np

def vec_gradient(X, Y, theta):
	m = X.shape[0]
	n= X.shape[1]
	return 1/m * np.transpose(X).dot(X.dot(theta) - Y)

def fit_(theta, X, Y, alpha,  n_cycle):
	m = X.shape[0]
	n = X.shape[1]
	new_theta = theta
	one = np.array([1 for i in range(m)]).reshape(m,1)
	X1 = np.hstack((one, X))

	for i in range(n_cycle):
		d_theta = vec_gradient(X1, Y, new_theta)
		new_theta = new_theta - alpha * d_theta
	return new_theta

if __name__ == "__main__":
	X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
	theta1 = np.array([[1.], [1.]])	
	th = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
	print(th)
