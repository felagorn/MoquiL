# 2022 Feralezer Tampubolon

import numpy as np

class AbstractActivationFunction:

    def __init__(self):
        raise NotImplementedError("Error: Cannot instantiate object from abstract class \"AbstractActivationFunction\"")

    def calculate(self, z: float):
        raise NotImplementedError("Error: Method \"calculate\" on parent class \"AbstractActivationFunction\" needs to be overriden by its child class")

    def calculate_derivative(self, z: float):
        raise NotImplementedError("Error: Method \"calculate_derivative\" on parent class \"AbstractActivationFunction\" needs to be overriden by its child class")

class Sigmoid(AbstractActivationFunction):

    def __init__(self):
        pass

    def calculate(self, z: float):
        if z >= 0:
            exp_z = np.exp(-z)
            return 1 / (1 + exp_z)
        else:
            exp_z = np.exp(z)
            return exp_z / (1 + exp_z)
    
    def calculate_derivative(self, z: float):
        return self.calculate(z) * (1 - self.calculate(z))

class Linear(AbstractActivationFunction):

    def __init__(self):
        pass

    def calculate(self, z: float):
        return z
    
    def calculate_derivative(self, z: float):
        return 1

class ReLU(AbstractActivationFunction):

    def __init__(self):
        pass

    def calculate(self, z: float):
        return max(0, z)
    
    def calculate_derivative(self, z: float):
        if z > 0:
            return 1
        else:
            return 0

class Softmax(AbstractActivationFunction):

    def __init__(self):
        pass
    
    # Problem: Underflowing if Z is too small
    # Trick: Shifting every element inside Z by max(Z)
    # because exp(X)/sum(exp(X)) = exp(X-max(X))exp(max(X))/sum(exp(X-max(X)))exp(max(X))
    def calculate(self, Z: list[float], Z_index: int):
        z_max = max(Z)
        return np.exp(Z[Z_index] - z_max) / sum([np.exp(Z[_] - z_max) for _ in range(len(Z))])
    
    def calculate_derivative(self, Z: list[float], Z_index: int, Y: list[int]):
        if (Y[Z_index] == 1):
            return -1 * (1 - Z[Z_index])
        elif (Y[Z_index] == 0):
            return Z[Z_index]
        else:
            raise ValueError("Error during softmax derivative calculation: Y needs to be one-hot encoded")