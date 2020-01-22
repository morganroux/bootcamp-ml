import math
import numbers 

def sigmoid_(x):
	if isinstance(x, numbers.Number):
		return 1 / (1 + math.exp(-x))
	return [1 / (1 + math.exp(-i)) for i in x]

x = 2
print(sigmoid_(x))
x = [-4, 2, 0]
print(sigmoid_(x))
