import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

def calc_WSS(X_data):
	WSS = []
	K = range(1,30)
	for k in K:
		kmean = KMeans(n_clusters = k).fit(X_data)
		WSS.append(kmean.inertia_)
	return K, WSS

def plot(K, WSS, name = 'WSS_K.png'):
	plt.plot(K, WSS, 'bx-')
	plt.xlabel('k')
	plt.ylabel('WSS')
	plt.title('Elbow Method for finding optimal k')
	plt.show()
	plt.savefig(name)
	
def main():
	with open (r'./pickle/X_data.pkl', 'rb') as file:
		X_data = pickle.load(file)
	K, WSS = calc_WSS(X_data)
	plot(K, WSS)
	
if __name__ == '__main__':
	main()