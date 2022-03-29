# 2022 Feralezer Tampubolon

from terminal import Terminal

class Neuron:

    def __init__(self, layer_id: int, neuron_id: int, use_bias = True)  -> None:
        self.layer_id = layer_id
        self.neuron_id = neuron_id
        self.use_bias = use_bias
        self.bias = Terminal()
        self.bias.set_forward_neuron(self)
        self.dendrites = []
        self.axons = []
        self.z_cache = []
    
    def add_dendrite(self, dendrite: Terminal) -> None:
        dendrite.set_forward_neuron(self)
        self.dendrites.append(dendrite)
    
    def add_axon(self, axon: Terminal) -> None:
        axon.set_backward_neuron(self)
        self.axons.append(axon)
    
    def get_bias(self) -> Terminal:
        return self.bias
    
    def get_dendrites(self) -> list[Terminal]:
        return self.dendrites

    def fire_forward(self, X: list[float]) -> float:
        if len(X) != len(self.dendrites):
            raise IndexError("Error in forward propagation (layer ID: " + str(self.layer_id) + ", neuron ID: " + str(self.neuron_id) + "): There needs to be an equal number of features and weights (excluding bias). Instead there are " + str(len(X)) + " features and " + str(len(self.dendrites)) + " weights (excluding bias).")
        else:
            Z = []
            if self.use_bias:
                Z.append(self.bias.propagate_forward(1))
            for i in range(len(X)):
                Z.append(self.dendrites[i].propagate_forward(X[i]))
            z = sum(Z)
            self.z_cache.append(z)
            return z
    
    def fire_backward(self, D: list[float]) -> float:
            if len(D) != len(self.axons):
                raise IndexError("Error in backward propagation (layer ID: " + str(self.layer_id) + ", neuron ID: " + str(self.neuron_id) + "): There needs to be an equal number of deltas and weights (excluding bias) from the next layer. Instead there are " + str(len(D)) + " deltas and " + str(len(self.axons)) + " next-layer weights (excluding bias).")
            else:
                Z = []
                for i in range(len(D)):
                    Z.append(self.axons[i].propagate_backward(D[i]))
                return sum(Z)
    
    def pop_z_cache(self) -> float:
        try:
            el = self.z_cache.pop()
            return el
        except IndexError:
            raise IndexError("pop from empty list on a neuron (layer ID: " + str(self.layer_id) + ", neuron ID: " + str(self.neuron_id) + ")")
    
    def get_z_cache_last_element(self) -> float:
        return self.z_cache[-1]
    
    def append_to_z_cache(self, z: float) -> None:
        self.z_cache.append(z)

    def clear_z_cache(self) -> None:
        self.z_cache.clear()