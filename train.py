import pytest
import numpy as np
from cnn import CNN
import os
from tensorflow.keras.datasets import cifar10

def test_evaluate():
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	number_of_train_samples_to_use = 500
	number_of_test_samples_to_use = 200
	X_train = X_train[0:number_of_train_samples_to_use, :]
	y_train = y_train[0:number_of_train_samples_to_use]
	X_test = X_test[0:number_of_test_samples_to_use,:]
	y_test = y_test[0:number_of_test_samples_to_use]
	my_cnn=CNN()
	my_cnn.add_input_layer(shape=(32,32,3),name="input")
	my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3,3),padding="same", activation='relu', name="conv1")
	my_cnn.append_maxpooling2d_layer(pool_size=(2,2),name="pool1")
	my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, activation='relu', name="conv2")
	my_cnn.append_maxpooling2d_layer(pool_size=(2,2),name="pool2")
	my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, activation='relu', name="conv3")
	my_cnn.append_flatten_layer(name="flat1")
	my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense1")
	my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="dense2")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv1")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv1")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv2")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv2")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv3")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv3")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="dense1")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="dense1")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="dense2")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="dense2")
	my_cnn.set_loss_function()
	my_cnn.set_optimizer(optimizer="SGD",learning_rate=0.01,momentum=0.0)
	my_cnn.set_metric(metric="accuracy")
	# los = np.array([2.30277, 2.30264, 2.30242, 2.30225, 2.30207, 2.30190, 2.30171, 2.30154, 2.30138])
	# los = np.around(los,decimals=2)
	my_cnn.train(X_train,y_train,60,10)
	acc = my_cnn.evaluate(X_test,y_test)
	de = np.float32(0.07)
	assert (acc == de)

def test_train():
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	number_of_train_samples_to_use = 500
	X_train = X_train[0:number_of_train_samples_to_use, :]
	y_train = y_train[0:number_of_train_samples_to_use]
	my_cnn=CNN()
	my_cnn.add_input_layer(shape=(32,32,3),name="input")
	my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3,3),padding="same", activation='relu', name="conv1")
	my_cnn.append_maxpooling2d_layer(pool_size=(2,2),name="pool1")
	my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, activation='relu', name="conv2")
	my_cnn.append_maxpooling2d_layer(pool_size=(2,2),name="pool2")
	my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, activation='relu', name="conv3")
	my_cnn.append_flatten_layer(name="flat1")
	my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense1")
	my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="dense2")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv1")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv1")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv2")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv2")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="conv3")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="conv3")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="dense1")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="dense1")
	np.random.seed(seed=1)
	weigh = my_cnn.get_weights_without_biases(layer_name="dense2")
	w_set = np.random.rand(*weigh.shape)
	my_cnn.set_weights_without_biases(w_set,layer_name="dense2")
	my_cnn.set_loss_function()
	my_cnn.set_optimizer(optimizer="SGD",learning_rate=0.01,momentum=0.0)
	my_cnn.set_metric(metric="accuracy")
	los = np.array([2.30277, 2.30264, 2.30242, 2.30225, 2.30207, 2.30190, 2.30171, 2.30154, 2.30138])
	los = np.around(los,decimals=2)
	hist = my_cnn.train(X_train,y_train,60,10)
	# print(hist['loss'])
	assert np.array_equal(los,np.around(hist['loss'][1:],decimals=2))
# return None
