# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample

        """
        self.model = keras.Sequential()
        self.loss = None
        self.metric = list()
        self.layers_number = []
        self.layers_name=[]


    def add_input_layer(self, shape=(2,),name="" ):
        # self.model.add(keras.Input(shape=shape,name=name))
        self.input_layer_tensor = shape
        self.input = keras.Input(shape=shape,name=name)
        self.model.add(self.input)
        self.layers_name.append(name)
        self.layers_number.append(-1)
        """
         This function adds a dense layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """



    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        if type(self.model) == type(keras.Model()):
            self.last_layer = self.model.output
            outputs = keras.layers.Dense(num_nodes,activation=activation,name=name,trainable=trainable)(self.last_layer)
            self.model = keras.Model(self.model.input,outputs)
            self.layers_number = [-1]+[i for i in range(len(self.model.layers)+1)]
        else:
            if type(name)==str and len(name) !=0:
                self.layers_name.append(name)
            self.layers_number.append(len(self.layers_number)-1) 
            self.model.add(keras.layers.Dense(num_nodes,activation=activation,name=name,trainable=trainable))

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,activation="Relu",name="",trainable=True):
        if type(name)==str and len(name) !=0:
            self.layers_name.append(name)
        self.layers_number.append(len(self.layers_number)-1) 
        return self.model.add(keras.layers.Conv2D(num_of_filters,kernel_size, strides=(1, 1), padding=padding, activation=activation, use_bias=True,trainable=trainable,name=name))
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        if type(name)==str and len(name) !=0:
            self.layers_name.append(name)
        self.layers_number.append(len(self.layers_number)-1) 
        return self.model.add(keras.layers.MaxPool2D(pool_size=pool_size,strides=strides,padding=padding,name=name))

    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        self.layers_name.append(name)
        self.layers_number.append(len(self.layers_number)-1) 
        return self.model.add(keras.layers.Flatten(name=name))

    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        if layer_number ==0:
            return None
        else:
            # return self.model.layers[self.layers_number[layer_number]].get_weights()[0]
            if layer_number != None:
                weights = self.model.layers[self.layers_number[layer_number]].get_weights()
                if len(weights)==0:
                    return None
                else:
                    return weights[0]
            elif len(layer_name) !=0:
                layr = self.model.get_layer(name=layer_name)
                weights = layr.get_weights()
                if len(weights)==0:
                    return None
                else:
                    return weights[0]
            else:
                return None
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """


    def get_biases(self,layer_number=None,layer_name=""):
        if layer_number ==0:
            return None
        else:
            # return self.model.layers[self.layers_number[layer_number]].get_weights()[0]
            if layer_number != None:
                weights = self.model.layers[self.layers_number[layer_number]].get_weights()
                if len(weights)==0:
                    return None
                else:
                    return weights[1]
            elif len(layer_name) !=0:
                layr = self.model.get_layer(name=layer_name)
                weights = layr.get_weights()
                if len(weights)==0:
                    return None
                else:
                    return weights[1]
            else:
                return None
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        if layer_number ==0:
            return None
        else:
            # return self.model.layers[self.layers_number[layer_number]].get_weights()[0]
            if layer_number != None:
                wei = self.model.layers[self.layers_number[layer_number]].get_weights()
                if len(wei)==0:
                    return None
                else:
                    wei[0]= weights
                    self.model.layers[self.layers_number[layer_number]].set_weights(wei)
                    return None
            elif len(layer_name) !=0:
                layr = self.model.get_layer(name=layer_name)
                wei = layr.get_weights()
                if len(wei)==0:
                    return None
                else:
                    wei[0]= weights
                    layr.set_weights(wei)
                    return None
            else:
                return None
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
    def set_biases(self,biases,layer_number=None,layer_name=""):
        # weights = self.model.layers[layer_number].get_weights()[0]
        # weights_bias = np.array(weights,biases)
        # self.model.layers[layer_number].set_weights(weights_bias)
        if layer_number ==0:
            return None
        else:
            # return self.model.layers[self.layers_number[layer_number]].get_weights()[0]
            if layer_number != None:
                wei = self.model.layers[self.layers_number[layer_number]].get_weights()
                if len(wei)==0:
                    return None
                else:
                    wei[1]= biases
                    self.model.layers[self.layers_number[layer_number]].set_weights(wei)
                    return None
            elif len(layer_name) !=0:
                layr = self.model.get_layer(name=layer_name)
                wei = layr.get_weights()
                if len(wei)==0:
                    return None
                else:
                    wei[1]= biases
                    layr.set_weights(wei)
                    return None
            else:
                return None
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
    def remove_last_layer(self):
        self.model=keras.Model(inputs=self.model.input,outputs=self.model.layers[-2].output)
        print(self.model.summary())
        self.model.layers.pop()
        print(self.model.summary())

    def pop_last_layer(self):
        """
        This function removes a layer from the model and connects the previous and next layer
        (if they exist).
        :return: poped layer
        """
        self.model=keras.Model(inputs=self.input_layer_tensor,outputs=self.model.layers[-2].output)
        return self.model.layers.pop()

    def load_a_model(self,model_name="",model_file_name=""):
        if len(model_name) != 0:
            if model_name == "VGG16":
                self.model = keras.applications.VGG16()
            elif model_name == "VGG19":
                self.model = keras.applications.VGG19()
        elif len(model_file_name)!=0:
            self.model = keras.models.load_model(model_file_name)
        self.layers_number = [-1]+[i for i in range(len(self.model.layers))]
        return self.model
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        
    def save_model(self,model_file_name=""):
        keras.models.save_model(self.model,model_file_name)
        return self.model
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        if loss == "SparseCategoricalCrossentropy":
            self.loss = keras.losses.SparseCategoricalCrossentropy()
        elif loss == "MeanSquaredError":
            self.loss = keras.losses.MeanSquaredError()
        elif loss == "hinge":
            self.loss = keras.losses.Hinge()
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """

    def set_metric(self,metric):
        self.metric.append(metric)
        # if metric == "mse":
        #     self.metric = keras.metrics.MeanSquaredError()
        
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        if optimizer =="SGD":
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        elif optimizer =="RMSprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate,momentum=momentum)
        elif optimizer == "Adagrad":
            self.optimizer = keras.optimizer.Adagrad(learning_rate=learning_rate)
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """

    def predict(self, X):
        return self.model.predict(x=X)
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """

    def evaluate(self,X,y):
        test_loss, test_acc = self.model.evaluate(x=X,y=y)
        return test_acc
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
    def train(self, X_train, y_train, batch_size, num_epochs):
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metric)
        hist = self.model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=num_epochs)
        return hist.history
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param X_validation: Array of input validation data
         :param y: Array of desired (target) validation outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    # print(my_cnn.model.summary())
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

    weights=my_cnn.get_weights_without_biases(layer_name="input")
    biases=my_cnn.get_biases(layer_number=0)
    print("input weights: ",None if weights is None else weights.shape,type(weights))
    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))
