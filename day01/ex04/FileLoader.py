import pandas as pd

class FileLoader:
	def __init__(self):
		pass
	def load(self, path):
		df = pd.read_csv(path)
		print("Loading dataset of dimensions " + " x ".join(tuple(str(x) for x in df.shape))) 
		return 	df
	
	def display(self, df, n): 
		pd.set_option('display.max_rows', None)
		print(df.head(n) if n > 0 else df.tail(-n))

if __name__ == "__main__":
		fl = FileLoader()
		df = fl.load("./resources/athlete_events.csv")
		fl.display(df, -10)
