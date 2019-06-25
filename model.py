import pickle
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.cluster import KMeans
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
from sklearn.preprocessing import normalize

#seeding
def init_centers(X_data, k):
	distances = [[] for x in X_data]	
	len_X_data = len(distances)
	#Return k rows of X
	rand_id = numpy.random.choice(len_X_data)
	first_center = X_data[rand_id]
	centers = first_center
	center = first_center
	for i in range (k-1):
		#Compute D^2(x) = distance between x and nearest center
		c = 0
		for x in X_data:
			vect = (x-center) * (x-center).T
			sum = vect.sum()
			distances[c].append(sum)
			c += 1
		D = [min(distances[x]) for x in range(len_X_data)]
		sum = 0
		for d in D:
			sum += d
		prob = [x/sum for x in D]
		rand_id = numpy.random.choice(len_X_data, p = prob)
		center = X_data[rand_id]
		centers = sparse.vstack([centers, center])
	with open(r'./pickle/centers_sparse.pkl', 'wb') as file:
		pickle.dump(centers, file)
	return centers 
	'''
	with open (r'./pickle/centers_sparse.pkl', 'rb') as file:
		centers_sparse = pickle.load(file)
	return centers_sparse
	'''

def assign_labels(X_data, centers):
	D = euclidean_distances(X_data, centers)
	return numpy.argmin(D, axis = 1)

def update_centers(X_data, labels, K):
	def means(Xk):
		sum_Xk = Xk[0]
		for i in range(1,len(Xk)):
			sum_Xk += Xk[i]
		return normalize(sum_Xk, norm='l1')

	centers = None
	for k in range(K):
		Xk = []
		for i in range(len(labels)):
			if labels[i] == k:
				Xk.append(X_data[i]) 
		center_k = means(Xk)
		centers = sparse.vstack([centers, center_k])
	return centers #center la 1 sparse matrix
	
def is_converged(centers, new_centers):
	# print ('shape of centers: {}, shape of new_centers: {}'.format (centers.get_shape(), new_centers.get_shape())) #(20, 109337)
	sub_matrix = centers - new_centers
	row_col = sub_matrix.nonzero()
	if len(row_col[0]) != 0:
		return False
	return True
	
def non_lib_Kmeans(X_data, K):
	centers = init_centers(X_data, K)
	while True:
		labels = assign_labels(X_data, centers)
		#print (len(labels)) #18828
		new_centers = update_centers(X_data, labels, K)
		if is_converged(centers, new_centers):
			break
		centers = new_centers
	return centers, labels
		
def main():
	K = 20
	with open (r'./pickle/y_test.pkl', 'rb') as file:
		y_test = pickle.load(file)
	'''
	with open (r'./pickle/X_data.pkl', 'rb') as file:
		X_data = pickle.load(file)
	# Library-based
	library_kmeans = KMeans(n_clusters=K, random_state=0).fit(X_data)
	with open(r'./pickle/library_kmeans.pkl', 'wb') as file:
		pickle.dump(library_kmeans, file)
	
	# Non-library-based
	non_lib_centers, non_lib_labels = non_lib_Kmeans(X_data, K)
	with open(r'./pickle/non_lib_centers.pkl', 'wb') as file:
		pickle.dump(non_lib_centers, file)
	with open(r'./pickle/non_lib_labels.pkl', 'wb') as file:
		pickle.dump(non_lib_labels, file)
	'''
	
	with open (r'./pickle/library_kmeans.pkl', 'rb') as file:
		library_kmeans = pickle.load(file)
	with open (r'./pickle/non_lib_centers.pkl', 'rb') as file:
		non_lib_centers = pickle.load(file)
	with open (r'./pickle/non_lib_labels.pkl', 'rb') as file:
		non_lib_labels = pickle.load(file)
	
	print ("Library-based Kmeans model's centers:")
	print (library_kmeans.cluster_centers_)
	#print ('{} x {}'.format(len(library_kmeans.cluster_centers_), len(library_kmeans.cluster_centers_[0])))
	print ("Non-library-based Kmeans model's centers:")
	print (non_lib_centers)
	#print (non_lib_centers.get_shape())
	lib_com_acc = completeness_score(y_test, library_kmeans.labels_)
	lib_hom_acc = homogeneity_score(y_test, library_kmeans.labels_)
	print ('Completeness accurancy of library-based Kmeans model = %.2f%%' %  (lib_com_acc*100)) # 53.55%
	print ('Homogeneity accurancy of library-based Kmeans model = %.2f%%' %  (lib_hom_acc*100)) # 43.68%
	non_lib_com_acc = completeness_score(y_test, non_lib_labels)
	non_lib_hom_acc = homogeneity_score(y_test, non_lib_labels)
	print ('Completeness accurancy of non-library-based Kmeans model = %.2f%%' %  (non_lib_com_acc*100)) # 58.84%
	print ('Homogeneity accurancy of non-library-based Kmeans model = %.2f%%' %  (non_lib_hom_acc*100)) # 57.03%
	
	
	
if __name__ == '__main__':
	main()