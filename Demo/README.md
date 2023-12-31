# Demo Constest

## A. One-hot-encoding 1

A dataset contains 10 features. Six of them are numeric, the rest four are categorical with 2, 3, 5 and 9 categories respectively. How many features will be in this dataset after one-hot-encoding?

__Answer:__
* __25__

## B. Linear 1-d

Choose all correct statements about the linear regression model $y = ax + b$

__Answer:__

* __A dummy (constant) model is a special case of linear regression__
* __Linear regression is not applicable when all predictors are the same__

## C. Linear regression GD

Let $X$ and $y$ be the feature matrix and the vector of targets. In the gradient descent algorithm the weights of linear regression model are updated by the rule 
$$w = w - \eta \nabla \mathcal{L}(w)$$

$\text{What is the formula for } \nabla \mathcal{L}(w)?$

__Answer:__

$\quad \nabla \mathcal{L}(w) = X^T(Xw - y)$

## D. Sigmoid' 2

Implement `SigmoidDerivative` function which receives the value of the sigmoid function at some point $x$ and returns $\sigma'(x)$


### Notes
Your function must be named `SigmoidDerivative`, receive and return `float` or `np.array` object.

__Answer:__

```python
import numpy as np
from typing import Union

def SigmoidDerivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)

```
