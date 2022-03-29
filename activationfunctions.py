# 2022 Feralezer Tampubolon

from numpy import ndarray, exp

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

    def calculate(self, z: float) -> float:
        if z >= 0:
            exp_z = exp(-z)
            return 1 / (1 + exp_z)
        else:
            exp_z = exp(z)
            return exp_z / (1 + exp_z)
    
    def calculate_derivative(self, z: float) -> float:
        return self.calculate(z) * (1 - self.calculate(z))

class Linear(AbstractActivationFunction):

    def __init__(self):
        pass

    def calculate(self, z: float) -> float:
        return z
    
    def calculate_derivative(self, z: float) -> float:
        return 1

class ReLU(AbstractActivationFunction):

    def __init__(self):
        pass

    def calculate(self, z: float) -> float:
        return max(0, z)
    
    def calculate_derivative(self, z: float) -> float:
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
    def calculate(self, Z: ndarray) -> ndarray:
        z_max = Z.max()
        exp_z = exp(Z - z_max)
        return exp_z / exp_z.sum()
    
    def calculate_derivative(self, Z: list[float], Z_index: int, Y: list[int]) -> float:
        if (Y[Z_index] == 1):
            return -1 * (1 - Z[Z_index])
        elif (Y[Z_index] == 0):
            return Z[Z_index]
        else:
            raise ValueError("Error during softmax derivative calculation: Y needs to be one-hot encoded")