import numpy as np

def sum_(x, f):
	if len(x.shape) != 1 or not callable(f):
		return None
	s = 0;
	for i in x:
		s += f(i)
	return s

if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(sum_(X, lambda x: x))
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(sum_(X, lambda x: x**2))
	
