from FileLoader import FileLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class MyPlotLib:
	def __init__(self):
		pass
	def print(self, data, features, **kwargs):		
		x = [data.loc[data[f].notnull(), f].values for f in features]
		fig, axes = plt.subplots(1, len(features), sharey=True, tight_layout=True)
		print(x)
		if len(features) >= 2:
			for i in range(len(features)):
				sns.distplot(x[i], ax=axes[i], **kwargs)
		else:
			sns.distplot(x[0], **kwargs)

		plt.show()
	def histogram(self, data, features):
		self.print(data, features, kde=False)
	def density(self, data, features):
		self.print(data, features, hist = False)
	def pair_plot(self, data, features):
		pass
	def box_plot(self, data, features):
		pass
	def scatter(self, data, features):
		ax = sns.scatterplot(x=features[0], y=features[1], data=data)
		plt.show()
		
if __name__ == "__main__":
	loader = FileLoader()
	data = loader.load("./resources/athlete_events.csv")
	mpl = MyPlotLib()
	mpl.histogram(data, ['Height', 'Weight'])
	mpl.density(data, ['Height', 'Weight'])
