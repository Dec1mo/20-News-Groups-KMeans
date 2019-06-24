import pickle
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.cluster import KMeans
import random

#seeding
def init_centers(X_data, k):
	#Return k rows of X
	first_center = random.choice(X)
	return centers
def assign_labels():

def update_centers():

def is_converged():

def non_lib_Kmeans():

def main():
	print ('Loading pickle files')
	with open (r'./pickle/X_data.pkl', 'rb') as file:
		X_data = pickle.load(file)
	with open (r'./pickle/y_test.pkl', 'rb') as file:
		y_test = pickle.load(file)
	library_kmeans = KMeans(n_clusters=20, random_state=0).fit(X_data)
	with open(r'./pickle/library_kmeans.pkl', 'wb') as file:
		pickle.dump(library_kmeans, file)
	lib_com_acc = completeness_score(y_test, library_kmeans.labels_)
	lib_hom_acc = homogeneity_score(y_test, library_kmeans.labels_)
	print ('Completeness accurancy of library-based Kmeans model = %.2f%%' %  (lib_com_acc*100))
	print ('Homogeneity accurancy of library-based Kmeans model = %.2f%%' %  (lib_hom_acc*100))
	#print ('Accurancy of non-library-based Kmeans model = %.2f%%' %  )
	
	
if __name__ == '__main__':
	main()