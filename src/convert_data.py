"""
============
Convert data
============


"""

import numpy as np
import csv

INITIAL_DATE = [1, 1, 2015]
label_mapping = {
	'S-B - intern (901106)' : 0,
	'ISO Zertifizierung smO (2990191)' : 1,
	'ISO Zertifizierung SWO Netz (2910667)' : 2,
	'E - intern (901107)' : 3,
	'E - MPM (2900041)' : 4,
	'E - smartTT' : 5
	'S-B - smartTT2.0 (901106)' :6
	'Noch nicht zugeordnet' : 7
	}

def load_data(filename):
	"""Loads each row of a .csvfile and puts them into a list.

	The output is a list containing rows of the input file as list 
	entries. Each entry is itself another list containing all the column
	entries of the file.

	Args:
		filename: The input file name as a string. Needs to be a .csv.

	Returns:
		A list containing the rows of the file.

	"""
	data = []
	with open('../res/'+filename+'.csv', newline='') as f:
		data_reader = csv.reader(f, delimiter=';')
		for row in data_reader:
			data.append(row)
		del data[0]
	return data

def convert_data(filename):
	"""Converts the data read from a file into a convenient numeric format.

	Calls the 'load_data' function in order to retrieve information from
	the given input .csv file. This data is then converted into a format
	that can be used later.

	Args:
		filename: The input file name as a string. Needs to be a .csv.

	Returns:

	"""
	data = load_data(filename)
	convert = np.zeros((len(data), len(data[0])))
	for i in range(len(data)):
		# Convert date
		date = data[i][0].split(str='.')
		convert[i,0] = (date[2] - INITIAL_DATE[2]) * 365
		convert[i,0] += (date[1] - INITIAL_DATE[1]) * 30
		convert[i,0] += (date[0] - INITIAL_DATE[0]) * 1

		# Convert begin
		convert[i,1] = data[i][1]

		# Convert ending
		convert[i,2] = data[i][2]

		# Convert label
		labels = data[i][3].split(str=',')
		for label in labels:
			convert[i,3] = label_mapping[label]
		

	print(convert.shape)

