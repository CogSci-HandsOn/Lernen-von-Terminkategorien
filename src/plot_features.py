"""
=============
Plot features
=============


"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition

import convert_data

def plot_all_features(features, names=None):
	"""Plots all the features in relation to each other.

	It creates a series of figures to display the correlation between
	each of the features.
	
	Args:
		features: An array of numerical features obtained by 
			'convert_data.get_features'.

		names: A list with label names corresponding to features. 
			It is used for plotting. Defaults to 'None'.
	"""
	num_features = features.shape[1]
	num_subplots = np.ceil(np.sqrt(num_features))
	print(features.shape)
	for i, f1 in enumerate(features.T):
		fig = plt.figure(i)
		if names != None:
			fig.suptitle(names[i]+' vs.')

		for j, f2 in enumerate(features.T):
			plt.subplot(num_subplots, num_subplots, j+1)
			plt.plot(f1, f2)
			if names != None:
				plt.title(names[j])
			else:
				plt.title('Feature {0}'.format(j))
		plt.show()

def plot_most_interesting_features(features, names=None, maximize=True, thresh=0.2):
	"""Plots the features with high or low correlation.

	Computes the covariance matrix (correlation coefficient) of the 
	features and plots either the most or least correlating of them. 
	Values near perfect correlation (1) and no correlation at all (0) 
	will be discarded according to 'thresh'.

	Args:
		features: An array of numerical features obtained by 
			'convert_data.get_features'. The covariance matrix will
			be computed on this.

		names: Label names of the features. Used to label the plots.

		maximize: If 'True' it will be searched for maximum correlation.
			Else it will be searched for minimum correlation.

		thresh: Threshold to cancel out not perfectly correlating 
			features.
	"""
	if maximize:
		neutral = 0
	else:
		neutral = 1
	
	C = abs(np.nan_to_num(np.corrcoef(features, rowvar=0)))
	np.place(C, C>1-thresh, neutral)
	np.place(C, C<0+thresh, neutral)

	for i in range(9):
		if maximize:
			indices = np.argmax(C)
		else:
			indices = np.argmin(C)
		indices = np.unravel_index(indices, C.shape)

		plt.subplot(3, 3, i+1)
		plt.plot(features[:,indices[0]], features[:,indices[1]])
		plt.xlabel(names[indices[0]])
		plt.ylabel(names[indices[1]])
		plt.title('Correlation coefficient: {}'\
			.format(np.round(C[indices], 3)))
		C[indices] = neutral
	plt.show()

def plot_principal_components(features, n_components=9):
	"""Plots the principal components of 'features'.

	Extracts the principal components of 'features' and plots the 9
	most significant of them. This function uses the implementation
	of 'sklearn.decomposition.PCA'.

	Args:
		features: An array of numerical features obtained by 
			'convert_data.get_features'. The principal components will
			be computed upon this.
	"""
	pca = sklearn.decomposition.PCA(n_components)
	# pca = sklearn.decomposition.PCA(n_components='mle')
	pcs = pca.fit_transform(features)
	for i in range(9):
		plt.subplot(3, 3, i+1)
		plt.plot(pcs[:,i])
	plt.show()
	print(pcs.shape)

features, names = convert_data.get_features('data')

# plot_all_features(features, names)
plot_most_interesting_features(features, names, False, thresh=0.05)
plot_principal_components(features)

