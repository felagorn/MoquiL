# 2022 Feralezer Tampubolon

from activationfunctions import Sigmoid, Linear, ReLU, Softmax
from neuron import Neuron
from terminal import Terminal
from random import shuffle, random, randint
from numpy import log
from json import dump

class FeedForwardNeuralNetwork:

    def __init__(self) -> None:
        self.model = []
        self.has_input_layer = False
        self.has_output_layer = False
    
    def add_layer(self, number_of_neurons: int, activation_function: str) -> None:
        if not self.has_input_layer:
            raise RuntimeError("Cannot add hidden layer to a model that has no input layer yet.")
        elif self.has_output_layer:
            raise RuntimeError("Cannot add hidden layer to a model that already has an output layer.")
        else:
            if activation_function.lower() == "sigmoid":
                self.model.append({"neurons": [Neuron(len(self.model), _) for _ in range(number_of_neurons)], "activation_function": Sigmoid()})
            elif activation_function.lower() == "linear":
                self.model.append({"neurons": [Neuron(len(self.model), _) for _ in range(number_of_neurons)], "activation_function": Linear()})
            elif activation_function.lower() == "relu":
                self.model.append({"neurons": [Neuron(len(self.model), _) for _ in range(number_of_neurons)], "activation_function": ReLU()})
            elif activation_function.lower() == "softmax":
                self.model.append({"neurons": [Neuron(len(self.model), _) for _ in range(number_of_neurons)], "activation_function": Softmax()})
            else:
                raise ValueError("Unknown activation function when trying to add a neural network layer")
            for neuron in self.model[-1]["neurons"]:
                for previous_layer_neuron in self.model[-2]["neurons"]:
                    terminal = Terminal(randint(-10, 10))
                    neuron.add_dendrite(terminal)
                    previous_layer_neuron.add_axon(terminal)
    
    def create_input_layer(self, number_of_inputs: int) -> None:
        if self.has_input_layer:
            raise RuntimeError("Cannot add input layer to a model that already has one.")
        else:
            self.model.append({"neurons": [Neuron(0, _, False) for _ in range(number_of_inputs)], "activation_function": Linear()})
            for neuron in self.model[-1]["neurons"]:
                neuron.add_dendrite(Terminal(1))
            self.has_input_layer = True
    
    def create_output_layer(self, number_of_outputs: int, activation_function="softmax") -> None:
        if self.has_output_layer:
            raise RuntimeError("Cannot add output layer to a model that already has one.")
        else:
            self.add_layer(number_of_outputs, activation_function)
            self.has_output_layer = True

    def predict(self, X: list) -> list:
        O = X
        for neuron_id in range(len(self.model[0]["neurons"])):
            self.model[0]["neurons"][neuron_id].append_to_z_cache(O[neuron_id])
        for layer in self.model[1:]:

            # Summation
            Z = []
            for neuron in layer["neurons"]:
                Z.append(neuron.fire_forward(O))

            # Activation
            if isinstance(layer["activation_function"], Softmax):
                O = [layer["activation_function"].calculate(Z, _) for _ in range(len(Z))]
            else:
                O = [layer["activation_function"].calculate(_) for _ in Z]
        return O

    def train(self, X: list[list], y: list[list], batch_size, learning_rate=0.1, error_threshold=0.001, epochs=1000):
        if len(X) != len(y):
            raise IndexError("X and y must have the same length")
        below_threshold = False
        epoch = 0
        while (not below_threshold) and (epoch < epochs):
            epoch_error = 0

            # Shuffle the training set
            training_set = list(zip(X, y, strict=True))
            shuffle(training_set)

            # Divide the training set into batches
            batches = [training_set[_:(_ + batch_size)] for _ in range(0, len(training_set), batch_size)]

            # Iterate through each batch
            for batch in batches:

                # Iterate through each item inside the batch
                for item in batch:

                    # Predict the item
                    O = self.predict(item[0])

                    # Calculate the error
                    if isinstance(self.model[-1]["activation_function"], Softmax):
                        epoch_error += (-1) * log(O[item[1].index(1)]) / len(training_set)
                    else:
                        for output_index in range(len(O)):
                            epoch_error += 0.5 * ((item[1][output_index] - O[output_index]) ** 2) / len(training_set)

                    # Propagate the prediction backward
                    D = O
                    if isinstance(self.model[-1]["activation_function"], Softmax):
                        D = [(O[_] - item[1][_]) for _ in range(len(self.model[-1]["neurons"]))]
                        for neuron in self.model[-1]["neurons"]:
                            neuron.pop_z_cache()
                    else:
                        D = [((O[_] - item[1][_]) * self.model[-1]["activation_function"].calculate_derivative(self.model[-1]["neurons"][_].pop_z_cache())) for _ in range(len(self.model[-1]["neurons"]))]
                    for neuron_id in range(len(self.model[-1]["neurons"])):
                        bias = self.model[-1]["neurons"][neuron_id].get_bias()
                        _ = bias.propagate_backward(D[neuron_id])
                    for layer in reversed(self.model[:-1]):
                        # Summation
                        Z = []
                        for neuron in layer["neurons"]:
                            Z.append(neuron.fire_backward(D))
                        
                        # Activation
                        if isinstance(layer["activation_function"], Softmax):
                            D = [Z[_] for _ in layer["neurons"]]
                        else:
                            D = [(Z[_] * layer["activation_function"].calculate_derivative(layer["neurons"][_].pop_z_cache())) for _ in range(len(layer["neurons"]))]
                        
                        # Pre-emptively propagate delta to bias
                        for neuron_id in range(len(layer["neurons"])):
                            bias = layer["neurons"][neuron_id].get_bias()
                            _ = bias.propagate_backward(D[neuron_id])
                
                # Update weights
                for layer in self.model[1:]:
                    for neuron in layer["neurons"]:
                        bias = neuron.get_bias()
                        bias.update_weight(learning_rate)
                        dendrites = neuron.get_dendrites()
                        for dendrite in dendrites:
                            dendrite.update_weight(learning_rate)
            
            print("Epoch " + str(epoch) + ", error=" + str(epoch_error))
            if epoch_error < error_threshold:
                below_threshold = True
            epoch += 1
    
    def export_model(self, filename="a.json"):
        exported_model = []
        for layer in self.model[1:]:
            exported_layer = {"neurons": [([neuron.get_bias().get_weight()] + [dendrite.get_weight() for dendrite in neuron.get_dendrites()]) for neuron in layer["neurons"]], "activation_function": ""}
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