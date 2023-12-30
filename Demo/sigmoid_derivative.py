import numpy as np
from typing import Union

def SigmoidDerivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)
