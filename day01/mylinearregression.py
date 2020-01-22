import numpy as np

class MyLinearRegression():
	def __init__(self, theta):
		self.theta = theta

	def vec_gradient(self, X, Y, th):
		m = X.shape[0]
		n= X.shape[1]
		return 1/m * np.transpose(X).dot(X.dot(th) - Y)
	
	def mse_(self, y, y_hat):
		return float(np.dot(np.transpose(y - y_hat), y - y_hat) / len(y))
	
	def predict_(self, X):
		n = X.shape[0]
		one = np.array([1 for i in range(n)]).reshape(n,1)
		X1 = np.hstack((one, X))
		return X1.dot(self.theta)
	
	def cost_elem_(self, X, Y):
		M = X.shape[0]
		Y_pred = self.predict_(X)
		return 0.5 / M * np.array([(y_pred - y)**2 for y_pred, y in zip(Y_pred, Y)])

	def cost_(self, X, Y):
		M = X.shape[0]
		Y_pred = self.predict_(X)
		diff = Y_pred - Y
		return 0.5 / M * np.transpose(diff).dot(diff)
	
	def fit_(self, X, Y, alpha,  n_cycle):
		m = X.shape[0]
		n = X.shape[1]
		new_theta = self.theta
		one = np.array([1 for i in range(m)]).reshape(m,1)
		X1 = np.hstack((one, X))

		for i in range(n_cycle):
			d_theta = self.vec_gradient(X1, Y, new_theta)
			new_theta = new_theta - alpha * d_theta
		self.theta = new_theta

if __name__ == '__main__':
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
	print(mylr.predict_(X))
	print(mylr.cost_elem_(X,Y))
	print(mylr.cost_(X,Y))
	mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
	print(mylr.theta)
	print(mylr.predict_(X))
	print(mylr.cost_elem_(X,Y))
	print(mylr.cost_(X,Y))
