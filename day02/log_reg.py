import numpy as np
import pandas as pd
import numbers
import math

class LogisticRegressionBatchGd:
	def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant', thetas = []):
		self.alpha = alpha
		self.max_iter = max_iter
		self.verbose = verbose
		self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
		self.thetas = thetas
		# Your code here (e.g. a list of loss for each epochs...)
		
	def sigmoid_(self, x):
		if isinstance(x, numbers.Number):
			return np.array( 1 / (1 + math.exp(-x)))
		return np.array([1 / (1 + math.exp(-i)) for i in x]).reshape(x.shape[0],1 )
	
	def vec_log_gradient_(self, x, y_true, y_pred):
		return np.transpose(x).dot(y_pred - y_true)
	
	def fit(self, x_train, y_train, thetas_start = []):
		"""
		Fit the model according to the given training data.
		Args:
			x_train: a 1d or 2d numpy ndarray for the samples
            y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
			self : object
            None on any error.
        Raises:
            This method should not raise any Exception.
		"""
		m = x_train.shape[0]
		n = x_train.shape[1] + 1
		one = np.array([1 for i in range(m)]).reshape(m,1)
		X1 = np.hstack((one, x_train))

		new_thetas = np.array([0 for i in range(n)]).reshape(n,1) if thetas_start == [] else thetas_start
		for i in range(self.max_iter):
			y_pred = self.predict(X1, new_thetas)
			new_thetas = new_thetas - self.alpha * self.vec_log_gradient_(X1, y_train, y_pred)

		self.thetas= new_thetas

	def predict(self, X, thetas):
		"""
        Predict class labels for samples in x_train.
        Arg:
            x_train: a 1d or 2d numpy ndarray for the samples
        Returns:
            y_pred, the predicted class label per sample.
            None on any error.
        Raises:
            This method should not raise any Exception.
		"""
		X_theta = X.dot(thetas)
		Y_pred = self.sigmoid_(X_theta)
		return Y_pred

	def score(self, x_train, y_train):
		"""
        Returns the mean accuracy on the given test data and labels.
        Arg:
            x_train: a 1d or 2d numpy ndarray for the samples
            y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
            Mean accuracy of self.predict(x_train) with respect to y_true
            None on any error.
        Raises:
            This method should not raise any Exception.
		"""
		return (self.predict(x_train,self.thetas) == y_test).mean()

if __name__ == '__main__':

	model = LogisticRegressionBatchGd()

	df_train = pd.read_csv('./resources/dataset/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
	x_train, y_train = np.array(df_train.iloc[1:150, 1:82]), df_train.iloc[1:150, 0].values
	y_train = y_train.reshape(y_train.shape[0],1)
	df_test = pd.read_csv('./resources/dataset/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
	x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0].values
	# We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
	model = LogisticRegressionBatchGd(alpha=0.01, max_iter=1500, verbose=True, learning_rate='constant')
	# We fit our model to our dataset and display the score for the train and test datasets
	print(y_train.shape, x_train.shape)
	model.fit(x_train, y_train)

