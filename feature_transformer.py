import pickle
import gensim
import re
from appos import appos
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

def clean_data(list_of_text):
	new_list_of_text = []
	lemmatizer = WordNetLemmatizer()
	set_stopwords = set(stopwords.words('english'))
	print ('Cleaning data...')
    #clear unsuitable characters
	for data in list_of_text:
		#Remove http
		data = re.sub(r'http\S+', " ", data)
		#Remove html tags
		data = re.sub(r'<.*?>', " ", data)
		#Remove all www.
		data = re.sub(r'www\.*?', " ", data)
		'''
		#Remove all .com
		data = re.sub(r'.*?\.com', " ", data)
		'''
		#Convert all text to lower cases:
		data = data.lower()
		#Remove all non characters a-z and A-Z
		data = re.sub(r'([^a-zA-Z\s]+)', " ", data)
		#Normalize appos and lemmatize words:
		words = data.split()
		data = [appos[word] if word in appos else lemmatizer.lemmatize(word) for word in words]
		data = [word for word in data if word not in set_stopwords and len(word) >= 2]
		data = " ".join(data)
		new_list_of_text.append(data)
	print ('Done cleaning')
	return new_list_of_text

def tfidf(list_of_text):
	print ('Turning into tfidf vectors')
	tfidf_vect = TfidfVectorizer() 
	tfidf_vect.fit(list_of_text)	
	print('Turned into tfidf vectors successfully')
	return tfidf_vect
	
def main():
	with open (r'./pickle/list_of_text.pkl', 'rb') as file:
			list_of_text = pickle.load(file)
			
	list_of_text = clean_data(list_of_text)
	
	tfidf_vect = tfidf(list_of_text)
	with open (r'./pickle/tfidf_vect.pkl', 'wb') as file:
			pickle.dump(tfidf_vect, file)
			
	X_data = tfidf_vect.transform(list_of_text)
	with open (r'./pickle/X_data.pkl', 'wb') as file:
			pickle.dump(X_data, file)
	
	print(X_data)
	
	
if __name__ == '__main__':
	main()