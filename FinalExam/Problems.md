# Final Exam

## A. Who is Softmax?

Choose all vectors which could be outputs of 
`Softmax` function

__Answer:__

* [0.5, 0.5] 
* [0.1, 0.2, 0.3, 0.4]
* [1]

## B. MAP = MLE

Consider a standard linear regression model

$y_i = x_i^T w + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2), \quad i = 1, \ldots, n.$

Under which condition MLE estimation of the parameters $w$ is almost identical to MAP estimation with Gaussian prior $p(w) = \mathcal{N}(w|0, \tau^2 I)$?

__Answer:__


$(\sigma \rightarrow +0)$ or $\tau \rightarrow +\infty$

## C. Activations
Select all fuctions $\psi$ which are used as activations in MLP.

__Answer:__

* $\psi(x) = \frac{1}{1+e^{-x}}$
  
* $\psi(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

* $\psi(x) = \max\{0, x\}$

* $\psi(x) = \max\{ax, 0\}, 0 < \alpha < 1$

## D. Pooling 3

Apply $3 \times 3$ max pooling with stride $3$ to the following matrix:

$$
\begin{pmatrix}
156 & 161 & 121 & 173 & 53 & 203 & 188 & 225 & 190 \\
59 & 151 & 120 & 255 & 114 & 165 & 162 & 236 & 56 \\
120 & 228 & 213 & 86 & 248 & 23 & 139 & 167 & 11 \\
51 & 64 & 43 & 1 & 246 & 75 & 216 & 117 & 237 \\
146 & 77 & 89 & 226 & 47 & 16 & 117 & 4 & 73 \\
38 & 146 & 221 & 232 & 63 & 25 & 91 & 113 & 194 \\
\end{pmatrix}
$$

Type elements of the result separated by a single space.

__Notes:__

For instance, if your answer is

$$
\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}
$$

submit

1 2 3 4 5 6 7 8 9

__Answer:__

228 255 236 221 246 237

### E. RL decision
In reinforcement learning, how do we call the decision strategy of an agent which tries to maximize total cumulative reward?

__Answer__:

* policy

## F. 1-NN proba

There are two samples in the dataset: $(\frac{1}{4}, \frac{1}{2})$ is attributed to positive class, $(\frac{3}{4}, \frac{3}{4})$ - to negative. Applying 1-NN method to a random point from $[0, 1]^2$, what is the probability of assigning positive class label to it?

__Notes:__

Round your answer up to four decimal places if necessary. For example, if your answer is 
$0.123456789$, submit $0.1235$.

__Answer:__

* 0.5625

## G. Gini information gain

The labels $y$ and values of two features $x_1$ and $x_2$ are shown in the table.

| $y$ |$x_1$|$x_2$|
|-----|-----|-----|
| 0   | 5   | 4   |
| 0   | 1   | 3   |
| 0   | -2  | 5   |
| 0   | -3  | -3  |
| 0   | −2  | 3   |
| 1   | 1   | -2  |
| 1   | 3   | 2   |
| 1   | 1   | -4  |
| 2   | 7   | -1  |
| 2   | 2   | 5   |

Building a decision tree model, one splits all samples by condition $[x_1 < 0 ]$. What is the information gain of such split if Gini impurity measure is used?

__Notes__

Round your answer up to four decimal places if necessary. For example, if your answer is $0.123456789$, submit $0.1235$.

__Answer:__

* 0.4571

## H. Backprop complexity

An MLP consists of $L$ linear layers interleaved with ReLU activations. Let $n_0$ and $n_L$ be dimensionalities of input and output respectively, $n_1, \ldots, n_{L-1}$ — sizes of hidden layers. Denote $N = \sum\limits_{i=0}^{L} n_i, \quad Q = \sum\limits_{i=0}^{L} n_i^2, \quad M = \sum\limits_{i=1}^{L} n_i n_{i-1}.$

What is the complexity of a single backward pass through such neural network if the batch size is  $B$?

1. $O(N)$
2. $O(Q)$
3. $O(M)$
4. $O(BN)$
5. $O(BQ)$
6. $O(BM)$
7. $O(B^2 N)$
8. $O(B^2 Q)$
9. $O(B^2 M)$

__Answer:__

* 6

## I. Classify it!

Which algorithm or model is commonly used for classification tasks? Choose all that apply.

__Answer:__

* Logistic regression 
* k-NN 
* Naive Bayes 
* Random forest
* MLP 
* CNN

## J. Moore&Penrose

The Moore-Penrose pseudo inverse to $\mathbf{X} \in \mathbb{R}^{m \times n}$ is a unique matrix $\mathbf{X}^+ \in \mathbb{R}^{n \times m}$ satisfying the following properties:

- $\mathbf{X}\mathbf{X}^+\mathbf{X} = \mathbf{X}$;
- $\mathbf{X}^+\mathbf{X}\mathbf{X}^+ = \mathbf{X}^+$;
- $(\mathbf{X}\mathbf{X}^+)^T = \mathbf{X}\mathbf{X}^+$;
- $(\mathbf{X}^+\mathbf{X})^T = \mathbf{X}^+\mathbf{X}$.

Select the *false* statement about pseudo inverse matrices.

1. If $\mathbf{X}$ is a square invertible matrix then $\mathbf{X}^+ = \mathbf{X}^{-1}$
2. If $\mathbf{X}$ has only one column $x$ which is a unit vector, then $\mathbf{X}^+$ consists of one row equal to $x^T$
3. If $\mathbf{X}$ has full column rank then $\mathbf{X}^+ = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$
4. If $\mathbf{X}$ has full row rank then $\mathbf{X}^+ = \mathbf{X}^T(\mathbf{X}\mathbf{X}^T)^{-1}$
5. If $\mathbf{X}$ has linearly independent columns then it requires $O(n^3 + n^2m)$ arithmetical operations to calculate $\mathbf{X}^+$
6. If $\mathbf{X}$ has linearly independent rows then $\mathbf{X}^+\mathbf{X} = I_n$
7. Given the dataset $\mathcal{D} = (\mathbf{X}, \mathbf{y})$, the optimal weights of linear regression model fit to MSE loss are ${\hat{w}} = \mathbf{X}^+\mathbf{y}$
8. If $\sigma_1 \geq \ldots \geq \sigma_r > 0$ are singular values of $\mathbf{X}$, then $\frac{1}{\sigma_1}, \ldots, \frac{1}{\sigma_r}$ are singular values of $\mathbf{X}^+$

__Answer:__

* 6 

## K. ROC AUC update

Given true labels $y_1, y_2, y_3, y_4, y_5$ (either $0$ or $1$) and predicted probabilities $\hat{y}_1, \hat{y}_2, \hat{y}_3, \hat{y}_4, \hat{y}_5$, one calculates ROC AUC to obtain $0.75$. Afterwards one more training sample $y_6$ came in along with prediction $\hat{y}_6$ (without changing any previous values). What is the maximum possible value of ROC AUC metric after this update?

__Notes:__

Round your answer up to four decimal places if necessary. For example, if your answer is 
$0.123456789$, submit $0.1235$.

__Answer:__

* 0.875


## L. Bagging -1

A dataset consists of 6 distinct samples $X_1, X_2, X_3, X_4, X_5, X_6$. In bagging we use bootstrap to obtain a new dataset $D = (X_{i1}, X_{i2}, X_{i3}, X_{i4}, X_{i5}, X_{i6}), 1 \leq i_k \leq 6$. What is $P(X_1 \notin D)$? Find this probability in two settings:

- the order of samples does matter; for instance, datasets $(X_2, X_2, X_3, X_3, X_4, X_4)$ and $(X_2, X_3, X_4, X_4, X_3, X_2)$ are different;
- the order does not matter: here datasets $(X_2, X_2, X_3, X_3, X_4, X_4)$ and $(X_2, X_3, X_4, X_4, X_3, X_2)$ are identical.

Submit the difference between the larger and the lower probability.

__Notes:__

Round your answer up to four decimal places if necessary. For example, if your answer is $0.123456789$, submit $0.1235$.

## M. Return

Implement the function

`def get_return(rewards: np.array, gamma: float)`
which calculates total cumulative reward with discounting factor gamma.

__Notes:__

It is guaranteed that the size of the input does not exceed $10^6$.

**Do not forget to** `import numpy as np`!

__Answer:__

```python
import numpy as np
from typing import Any


def get_return(rewards: np.array, gamma: float) -> Any:
    T = len(rewards)
    time_steps = np.arange(T)
    discount_factors = gamma ** time_steps
    cumulative_return = np.sum(rewards * discount_factors)
    return cumulative_return
```

## N. Multi-log-reg-grad

Implement `MultiLogRegGrad` function which receives a feature matrix $\mathbf{X}$, vector of labels 
$\mathbf{y}$, matrix of weights $\mathbf{W}$. The function should return the gradient of multinomial logistic regression model (without regularization). All labels belong to interval $[0, K - 1]$ where $K -$ number of classes.

__Notes:__

Your function must be called MultiLogRegGrad with signature

```python
def MultiLogRegGrad(X: np.array, y: np.array, W: np.array) -> np.array:
```

**Do not forget to import required libraries such as** `numpy`!

__Answer:__

```python
import numpy as np
from scipy.special import expit

def MultiLogRegGrad(X: np.array, y: np.array, W: np.array) -> np.array:
    K = W.shape[1]
    y_onehot = np.eye(K)[y]
    
    grad = X.T.dot(expit(X.dot(W)) - y_onehot) 

    return grad
```


