from dataset import *
import matplotlib.pyplot as plt
from Net import *

#pytorch RNN regression: https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import matplotlib

torch.manual_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
	# remove Travis CI Error
	matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
	''' Min Max Normalization

	Parameters
	----------
	data : numpy.ndarray
		input data to be normalized
		shape: [Batch size, dimension]

	Returns
	----------
	data : numpy.ndarry
		normalized data
		shape: [Batch size, dimension]

	References
	----------
	.. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

	'''
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
	# noise term prevents the zero division
	return numerator / (denominator + 1e-7)


# train Parameters
learning_rate = 1
num_epochs = 500
input_size = 9
hidden_size = 200
num_classes = 1
timesteps = seq_length = 7
num_layers = 1  # number of layers in RNN


#Create the dataset
dataset = Dataset()
dataset.create()

X_train, Y_train = dataset.from_listdict_to_array(dataset.train_data, 'fare_amount')
Y_train = MinMaxScaler(Y_train.reshape(len(Y_train), 1))
print(Y_train)
X_val, Y_val = dataset.from_listdict_to_array(dataset.validation_data, 'fare_amount')
X_train = Variable(torch.Tensor(MinMaxScaler(X_train)))
X_val = Variable(torch.Tensor(MinMaxScaler(X_val)))
Y_train = Variable(torch.Tensor(MinMaxScaler(Y_train)))
Y_val = Variable(torch.Tensor(MinMaxScaler(Y_val)))

# Instantiate RNN model
net = Net(num_classes, input_size, hidden_size, num_layers)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
	outputs = net(X_train)
	optimizer.zero_grad()
	# obtain the loss function
	loss = criterion(outputs, Y_train)
	loss.backward()
	optimizer.step()
	print("Epoch: %d, loss: %1.5f" % (epoch, loss.data[0]))

print("Learning finished!")

'''
plot_list = []
plot_list.append([preu['year'] for preu in dataset.train_data])
plot_list.append([preu['month'] for preu in dataset.train_data])
print(dataset.train_data[0])
plt.scatter(plot_list[0], plot_list[1], s=2, alpha=0.5)
plt.show()
'''