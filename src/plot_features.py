"""
=============
Plot features
=============


"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D

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
		if names is not None:
			fig.suptitle(names[i]+' vs.')

		for j, f2 in enumerate(features.T):
			plt.subplot(num_subplots, num_subplots, j+1)
			plt.plot(f1, f2)
			if names is not None:
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
		if names is not None:
			plt.xlabel(names[indices[0]])
			plt.ylabel(names[indices[1]])
		plt.title('Correlation coefficient: {}'\
			.format(np.round(C[indices], 3)))
		C[indices] = neutral
	plt.show()

def plot_principal_components(features, labels, n_components=3):
	"""Plots the principal components of 'features'.

	Extracts the principal components of 'features' and plots the 9
	most significant of them. This function uses the implementation
	of 'sklearn.decomposition.PCA'.

	Args:
		features: An array of numerical features obtained by 
			'convert_data.get_features'. The principal components will
			be computed upon this.
	"""
	## Perform PCA
	pca = sklearn.decomposition.PCA(n_components)
	features = sklearn.preprocessing.normalize(features, 'max', 0)
	pcs = pca.fit(features)
	mapping = pca.transform(features)


	## Map on three PCs of highest eigenvalues
	fig = plt.figure(1)
	fig.suptitle('Features mapped on PCs with largest $\lambda$')
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(mapping[:,0], mapping[:,1], mapping[:,2], c=labels)
	plt.show()


	## Maximize distance of labels
	print('Maximize distance of labels:')
	axes_combs = list(itertools.combinations(range(n_components), 3))
	X = np.empty((len(features), 3))
	labels = np.array(labels)
	fig = plt.figure(3)
	fig.suptitle('Plots mapping on PCs to seperate label 0-7 from others')
	for label in set(labels):
	# for label in {0}:
		print('  Computing best fit for label {}...'.format(label), 
			end='', flush=True)
		idx_in = np.where(label==labels)
		idx_out = np.where(label!=labels)

		best_dist = 0
		# Iterate through all axis combinations and try to maximize
		# distance of current label to other observations
		for axes_i in axes_combs:
			X[:,0] = mapping[:,axes_i[0]]
			X[:,1] = mapping[:,axes_i[1]]
			X[:,2] = mapping[:,axes_i[2]]

			# Compute distance matrix of all observations
			v = scipy.spatial.distance.pdist(X)
			dist = scipy.spatial.distance.squareform(v)

			# Compute mean distance of label observations and
			# weight it according to total number of observations
			mean = np.mean(
				[np.mean(dist[r,idx_out[0]]) for r in idx_in[0]])
			dist_tmp = mean * len(idx_in) / len(features)
			
			# Normalize measure with mean distance of all observations
			dist_tmp /= 0.5 * np.mean(dist)
 
			# Check for new maximal distance of current label
			if dist_tmp > best_dist:
				best_dist = dist_tmp
				axes_of_choice = axes_i
		ax = fig.add_subplot(3, 3, label+1, projection='3d')
		ax.set_title('Label {}'.format(label))
		ax.scatter(
			mapping[:,axes_of_choice[0]], 
			mapping[:,axes_of_choice[1]], 
			mapping[:,axes_of_choice[2]], 
			c=labels, label=label)
		ax.set_xlabel(axes_of_choice[0])
		ax.set_ylabel(axes_of_choice[1])
		ax.set_zlabel(axes_of_choice[2])
		plt.legend()
		print(' done!')
	plt.show()

	
features, labels, names = convert_data.get_features(
	'2016_05_09     HandsOn CogSci ___ Daten f\\303\\274r'
	' Lernen von Terminkategorien')

# plot_all_features(features, names)
# plot_most_interesting_features(features, names, False, thresh=0.05)
# plot_principal_components(features, labels, 50)
