import numpy as np
from FileLoader import FileLoader as FL
from MyPlotLib import MyPlotLib as MyPL
from mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fl = FL()
mypl= MyPL()
data = fl.load('./resources/are_blue_pills_magics.csv')
mypl.scatter(data, ('Micrograms', 'Score')
