import numpy as np
from FileLoader import FileLoader as FL
from MyPlotLib import MyPlotLib as MyPL
from mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Init
fl = FL()
mypl= MyPL()
data = fl.load('../resources/are_blue_pills_magics.csv')
ax = sns.scatterplot(x='Micrograms', y='Score', data=data)
mylr = MyLR([[60.0], [-1]])
X = np.array(data.loc[:, 'Micrograms'].values)
X = X.reshape(X.shape[0],1)
Y = np.array(data.loc[:, 'Score'].values)
Y = Y.reshape(Y.shape[0],1)

# Fitting
mylr.fit_(X,Y,0.01,2000)
print(mylr.theta)

# Plotting result
Y_pred = mylr.predict_(X)
data2 = {
	'Micrograms': data.loc[:, 'Micrograms'].values,
	'Predict' : mylr.predict_(X).reshape(X.shape[0])
}	
ax = sns.lineplot(x='Micrograms', y='Predict',data=data2, ax=ax)
plt.show()

