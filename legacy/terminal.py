# 2022 Feralezer Tampubolon

from random import randint

class Terminal:

    def __init__(self, weight=0.0) -> None:
        self.weight = weight
        self.forward_neuron = None
        self.backward_neuron = None
        self.forward_cache = []
        self.backward_cache = []
    
    def set_forward_neuron(self, neuron) -> None:
        self.forward_neuron = neuron
    
    def set_backward_neuron(self, neuron) -> None:
        self.backward_neuron = neuron
    
    def clear_cache(self) -> None:
        self.forward_cache.clear()
        self.backward_cache.clear()
    
    def propagate_forward(self, x) -> float:
        self.forward_cache.append(x)
        return self.weight * x
    
    def propagate_backward(self, d) -> float:
        self.backward_cache.append(d)
        return self.weight * d
    
    def get_forward_cache_average(self) -> float:
        return sum(self.forward_cache) / len(self.forward_cache)
    
    def get_backward_cache_average(self) -> float:
        return sum(self.backward_cache) / len(self.backward_cache)
    
    def update_weight(self, learning_rate: float) -> None:
        self.weight += (-1) * learning_rate * (sum(self.backward_cache) / len(self.backward_cache)) * (sum(self.forward_cache) / len(self.forward_cache))
        self.clear_cache()
    
    def get_weight(self) -> float:
        return self.weight