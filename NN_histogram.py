#!/usr/bin/env python
# coding: utf-8


"""Classifying histograms with a Neural Network.  

to explore : http://www.chioka.in/why-is-keras-running-so-slow/


"""


TEST_SIZE           =    10 
POURCENTAGE_TRAIN   =    0.15

HIDDEN_LAYER_SIZE = 55 
NB_FEATURES       = 111


from sklearn import cross_validation 
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd 
import numpy as np

class Data: 
	def __init__(self, path): 
		"""we need to saturate. """
		self.file_name     = path
		self.loaded_frame  = pd.read_csv(self.file_name, sep=',',header=None) 
		self.frame_numpied = self.loaded_frame.values 
		self.dim_x         = (self.frame_numpied.shape[0], self.frame_numpied.shape[1] - 1)
		self.dim_y         = (self.frame_numpied.shape[0], 1)
		self.X             = self.frame_numpied[ :self.dim_x[0], :self.dim_x[1]]
		self.y             = self.frame_numpied[ :self.dim_y[0], 111]
		self.X_app, self.X_test, self.y_app, self.y_test = self.cross_validate()
		
	def cross_validate(self): 
		""" needs to change to this https://stackoverflow.com/questions/25889637/how-to-use-k-fold-cross-validation-in-a-neural-network """
		return cross_validation.train_test_split(self.X, self.y, test_size =TEST_SIZE, train_size=POURCENTAGE_TRAIN, random_state=0)
		#print(X_app)
		
class NeuralNetwork:
	def __init__ (self, x, y): 
		self.X_train = x
		self.Y_train = y 
		self.NN      = self.build()
				
	def build(self): 
		Network = Sequential() 
		Network.add(Dense(output_dim = HIDDEN_LAYER_SIZE, init='uniform', activation='relu', input_dim=NB_FEATURES)) #first hidden layer
		Network.add(Dense(output_dim = 1, init='uniform', activation='sigmoid')) #output neuron 
		Network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #compile NN
			#optimizer : how to tune wieghts
			#loss      : loss function to optimize  (exemple : sum(y - ŷ)^^2)
			#metrics   : accuracy
		return Network
	
	def fit_to_training(self):
		self.NN.fit(self.X_train, self.Y_train, batch_size = 10, nb_epoch = 10000 )
		#self.NN.predict(np.transpose(self.simulate()))
		self.visualize()

	def simulate(self): 
		return np.random.randint(low=0, high=5, size=111, dtype='l').T
	
	def visualize(self):
		fig = plt.figure(figsize=(12, 12))
		self.draw_neural_net(fig.gca(), .1, .9, .1, .9, [20, 7, 1])
		plt.show()

	def draw_neural_net(self, ax, left, right, bottom, top, layer_sizes):
		n_layers = len(layer_sizes)
		v_spacing = (top - bottom)/float(max(layer_sizes))
		h_spacing = (right - left)/float(len(layer_sizes) - 1)
		# Nodes
		for n, layer_size in enumerate(layer_sizes):
			layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
			for m in range(layer_size):
				circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
									color='w', ec='k', zorder=4)
				ax.add_artist(circle)
		# Edges
		for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
			layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
			for m in range(layer_size_a):
				for o in range(layer_size_b):
					line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
									[layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
					ax.add_artist(line)
	


if __name__ == "__main__": 
	NeuralNetwork(Data('dataset.csv').X, Data('dataset.csv').y).fit_to_training()

	






