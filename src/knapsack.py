from itertools import product
from typing import Tuple
import time

import numpy as np

NMIN = 10
NMAX = 40000
RAND_MAX = 1000
array = np.ndarray


def create_knapsack_instance(N: int, divide=6) -> Tuple[float, array, array]:
    N = max(NMIN, min(N, NMAX))
    # Use Mersenne Twister
    values = np.array([RAND_MAX * np.random.rand() + 1 for i in range(N)])
    # Use Mersenne Twister
    weights = np.array([RAND_MAX * np.random.rand() + 1 for i in range(N)])

    ratio = values / weights
    values = values[np.argsort(-ratio)]
    weights = weights[np.argsort(-ratio)]
    ratio = ratio[np.argsort(-ratio)]

    capacity = RAND_MAX * N / divide

    return capacity, values, weights


class KnapsackSolover:
    """ Abstract class of Knapsack Problem Solover. """

    def __init__(self, capacity: int, values: array, weights: array) -> None:
        self.capacity = capacity
        self.values = values
        self.weights = weights
        self.N = len(values)
        self.check_N()
        self.x_opt = None
        self.v_opt = None

    def check_N(self) -> None:
        if self.N != self.weights.shape[0]:
            raise ValueError("Length of values and weights is not same.")

    def solve(self) -> None:
        return self.x_opt


class KnapsackEnumerationSolover(KnapsackSolover):
    """ Class of Knapsack Problem Solover by Enumeration. """

    def solve(self) -> Tuple[array, float, float]:
        self.v_opt = 0
        self.x_opt = np.zeros(self.N)
        for x in product(*[[0, 1] for i in range(self.N)]):
            x = np.array(x)
            sum_value = (x * self.values).sum()
            sum_weight = (x * self.weights).sum()
            if self.v_opt < sum_value and sum_weight < self.capacity:
                self.v_opt = sum_value
                self.x_opt = x

        w_opt = (self.x_opt * self.weights).sum()
        return self.x_opt, self.v_opt, w_opt


class KnapsackGreedySolover(KnapsackSolover):
    """ Class of Knapsack Problem Solover by Greedy Algorithm. """

    def solve(self) -> Tuple[array, float, float]:
        sum_weight = 0
        self.v_opt = 0
        self.x_opt = np.zeros(self.N)

        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            if sum_weight + w > self.capacity:
                continue
            else:
                sum_weight += w
                self.v_opt += v
                self.x_opt[i] = 1

        return self.x_opt.astype(int), self.v_opt, sum_weight


class KnapsackRelaxedLinearSolover(KnapsackSolover):
    """ Class of Knapsack Problem Solover by Relaxed Linear optimization. """

    def solve(self) -> Tuple[array, float, float]:
        sum_weight = 0
        self.v_opt = 0
        self.x_opt = np.zeros(self.N)

        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            if sum_weight + w > self.capacity:
                ratio = (self.capacity - sum_weight) / w
                sum_weight = int(sum_weight + w * ratio)
                self.v_opt = int(self.v_opt + v * ratio)
                self.x_opt[i] = ratio
                break
            else:
                sum_weight += w
                self.v_opt += v
                self.x_opt[i] = 1

        return self.x_opt, self.v_opt, sum_weight


def main():
    print("N = ", end="")

    N = int(input())
    capacity, values, weights = create_knapsack_instance(N)

    print(f"capacity = {capacity}")
    print(f"values = {values}")
    print(f"weight = {weights}")
    print(f"ratio = {values / weights}")

    print("Enumeration")
    KES = KnapsackEnumerationSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KES.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")

    print("Greedy")
    KGS = KnapsackGreedySolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KGS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")

    print("Relaxed Linear")
    KRLS = KnapsackRelaxedLinearSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KRLS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")


if __name__ == "__main__":
    main()
