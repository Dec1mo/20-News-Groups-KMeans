import os
import io
import pickle

def get_all_file(path):
	list_of_file_path = []
	true_label = []
	current_label = 0
	for path, subdirs, files in os.walk(path):
		for name in files:
			list_of_file_path.append(os.path.join(path, name))
			true_label.append(current_label)
		current_label += 1
	return list_of_file_path, true_label

def file_to_text_list(list_of_file_path):
	list_of_text = []
	for file in list_of_file_path:
		with open(file, 'r+') as f:
			text = f.read()
			list_of_text.append(text)
	return list_of_text
	
def main():
	path = '20news-18828'
	list_of_file_path, y_test = get_all_file(path)
	list_of_text = file_to_text_list(list_of_file_path)
	
	print('Pickling')
	with open(r'./pickle/list_of_text.pkl', 'wb') as file:
		pickle.dump(list_of_text, file)
		
	with open(r'./pickle/y_test.pkl', 'wb') as file:
		pickle.dump(y_test, file)
	print('Done pickling')
	
if __name__ == '__main__':
	main()
	