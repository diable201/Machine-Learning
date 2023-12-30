import numpy as np
from typing import Any


def get_return(rewards: np.array, gamma: float) -> Any:
    T = len(rewards)
    time_steps = np.arange(T)
    discount_factors = gamma ** time_steps
    cumulative_return = np.sum(rewards * discount_factors)
    return cumulative_return
