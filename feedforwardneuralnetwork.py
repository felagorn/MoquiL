# 2022 Feralezer Tampubolon

from activationfunctions import Sigmoid, Linear, ReLU, Softmax
from numpy import ndarray, array, dot, log
from random import random, shuffle
from json import dump, load

class FeedForwardNeuralNetwork:

    def __init__(self) -> None:
        self.model = []

    def create_layer(self, number_of_neurons: int, activation_function: str, number_of_inputs: int) -> None:
        layer = {"neurons": array([array([random() for _ in range(number_of_inputs + 1)]) for _ in range(number_of_neurons)]), "activation_function": ""}
        if activation_function.lower() == "sigmoid":
            layer["activation_function"] = Sigmoid()
        elif activation_function.lower() == "linear":
            layer["activation_function"] = Linear()
        elif activation_function.lower() == "relu":
            layer["activation_function"] = ReLU()
        elif activation_function.lower() == "softmax":
            layer["activation_function"] = Softmax()
        else:
            raise NotImplementedError("Unknown activation function " + activation_function + " when trying to add a neural network layer")
        self.model.append(layer)
    
    def add_layer(self, number_of_neurons: int, activation_function: str) -> None:
        if len(self.model) == 0:
            raise RuntimeError("Cannot add hidden layer to a model that has no input layer yet. Create an input layer by calling .create_layer()")
        else:
            self.create_layer(number_of_neurons, activation_function, len(self.model[-1]["neurons"]))
    
    def __predict(self, X: ndarray) -> ndarray:
        O = [X]
        Z = []
        for layer in self.model:

            # Summation
            Z.append(array([dot(neuron, array([1] + O[-1].tolist())) for neuron in layer["neurons"]]))

            # Activation
            if isinstance(layer["activation_function"], Softmax):
                O.append(layer["activation_function"].calculate(Z[-1]))
            else:
                O.append(array([layer["activation_function"].calculate(z) for z in Z[-1]]))
        
        return array(Z), array(O)
    
    def predict(self, X: ndarray) -> ndarray:
        _, O = self.__predict(X)
        return O[-1]
    
    def train(self, X_train: ndarray, Y_train: ndarray, batch_size: int, learning_rate=0.1, error_threshold=0.001, epochs=1000) -> None:
        if len(X_train) != len(Y_train):
            raise IndexError("X_train and Y_train must have the same length")
        below_threshold = False
        epoch = 0
        while (not below_threshold) and (epoch < epochs):
            epoch_error = 0

            # Shuffle the training set
            training_set = list(zip(X_train, Y_train, strict=True))
            shuffle(training_set)

            # Divide the training set into batches
            batches = [training_set[_:(_ + batch_size)] for _ in range(0, len(training_set), batch_size)]

            # Iterate through each batch
            for batch in batches:

                # Iterate through each item in the batch
                batch_O = []
                batch_D = []
                for item in batch:

                    # Predict the item
                    Z, O = self.__predict(item[0])

                    # Calculate the error
                    if isinstance(self.model[-1]["activation_function"], Softmax):
                        epoch_error += (-1) * log(O[-1][item[1].index(1)]) / len(training_set)
                    else:
                        epoch_error += 0.5 * ((O[-1] - item[1]) ** 2).sum() / (len(training_set) * len(item[1]))
                    
                    # Propagate the prediction backward

                    # Output layer

                    # Activation
                    D = []
                    if isinstance(self.model[-1]["activation_function"], Softmax):
                        D.insert(0, O[-1] - item[1])
                    else:
                        D.insert(0, (O[-1] - item[1]) * array([self.model[-1]["activation_function"].calculate_derivative(Z[-1][_]) for _ in range(len(self.model[-1]["neurons"]))]))

                    # Hidden layers
                    for layer_id in reversed(range(len(self.model[:-1]))):

                        # Summation
                        S = array([dot(array([neuron[_ + 1] for neuron in self.model[layer_id + 1]["neurons"]]), D[-1]) for _ in range(len(self.model[layer_id]["neurons"]))])
                        
                        # Activation
                        if isinstance(self.model[layer_id]["activation_function"], Softmax):
                            D.insert(0, S)
                        else:
                            D.insert(0, S * array([self.model[layer_id]["activation_function"].calculate_derivative(Z[layer_id][_]) for _ in range(len(self.model[layer_id]["neurons"]))]))
                    
                    # Save calculations
                    batch_O.append(O)
                    batch_D.append(array(D))
                
                # Update weights
                for layer_id in range(len(self.model)):
                    for neuron_id in range(len(self.model[layer_id]["neurons"])):
                        average_d = array([item[layer_id][neuron_id] for item in batch_D]).sum() / len(batch_D)
                        for weight_id in range(len(self.model[layer_id]["neurons"][neuron_id])):
                            if weight_id == 0:
                                self.model[layer_id]["neurons"][neuron_id][weight_id] += (-1) * learning_rate * average_d
                            else:
                                average_x = array([item[layer_id][weight_id - 1] for item in batch_O]).sum() / len(batch_O)
                                self.model[layer_id]["neurons"][neuron_id][weight_id] += (-1) * learning_rate * average_d * average_x

            # Check total error
            print("Epoch " + str(epoch) + ", error=" + str(epoch_error), end="\r", flush=True)
            if epoch_error < error_threshold:
                below_threshold = True
            epoch += 1
    
    def export_model(self, filename="a.json") -> None:
        exported_model = []
        for layer in self.model:
            exported_layer = {"neurons": [neuron.tolist() for neuron in layer["neurons"]], "activation_function": ""}
            if isinstance(layer["activation_function"], Sigmoid):
                exported_layer["activation_function"] = "sigmoid"
            elif isinstance(layer["activation_function"], Linear):
                exported_layer["activation_function"] = "linear"
            elif isinstance(layer["activation_function"], ReLU):
                exported_layer["activation_function"] = "relu"
            elif isinstance(layer["activation_function"], Softmax):
                exported_layer["activation_function"] = "softmax"
            exported_model.append(exported_layer)
        with open(filename, "w") as f:
            dump(exported_model, f)
    
    def import_model(self, filename: str) -> None:
        imported_model = None
        with open(filename, "r") as f:
            imported_model = load(f)
        self.model.clear()
        for imported_layer in imported_model:
            layer = {"neurons": array([array(imported_neuron) for imported_neuron in imported_layer["neurons"]]), "activation_function": ""}
            if imported_layer["activation_function"].lower() == "sigmoid":
                layer["activation_function"] = Sigmoid()
            elif imported_layer["activation_function"].lower() == "linear":
                layer["activation_function"] = Linear()
            elif imported_layer["activation_function"].lower() == "relu":
                layer["activation_function"] = ReLU()
            elif imported_layer["activation_function"].lower() == "softmax":
                layer["activation_function"] = Softmax()
            else:
                raise NotImplementedError("Unknown activation function " + imported_layer["activation_function"] + " when trying to add a neural network layer")
            self.model.append(layer)