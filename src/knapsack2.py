import time
from typing import Tuple

import numpy as np

from knapsack import KnapsackSolover, create_knapsack_instance

array = np.ndarray


def get_fixed_instance(capacity, values, weights, fix_indices):
    fixed_value = values[fix_indices].sum()
    fixed_weight = weights[fix_indices].sum()

    fixed_x = np.zeros(values.shape[0])
    fixed_x[fix_indices] = 1

    indices = np.arange(values.shape[0])
    nonfix_indices = indices[indices != fix_indices]

    nonfix_capacity = capacity - fixed_weight
    nonfix_values = values[nonfix_indices]
    nonfix_weights = weights[nonfix_indices]

    return nonfix_capacity, nonfix_values, nonfix_weights, fixed_value


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

    def fixed_solve(self, fix_indices: array) -> Tuple[array, float, float]:
        self.x_opt = np.zeros(self.N)
        self.x_opt[fix_indices] = 1
        self.v_opt = self.values[fix_indices].sum()
        sum_weight = self.weights[fix_indices].sum()

        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            if sum_weight + w > self.capacity or i in fix_indices:
                continue
            else:
                sum_weight += w
                self.v_opt += v
                self.x_opt[i] = 1

        return self.x_opt.astype(int), self.v_opt, sum_weight


class KnapsackHalfAproxSolover(KnapsackSolover):
    def solve(self) -> Tuple[array, float, float]:
        child_solver = KnapsackGreedySolover(
            self.capacity, self.values, self.weights)
        x_opt1, v_opt1, sum_weight1 = child_solver.solve()
        fixidx = np.where(x_opt1 == 0)[0].min()
        x_opt2, v_opt2, sum_weight2 = child_solver.fixed_solve(
            np.array([fixidx]))

        if v_opt1 >= v_opt2:
            self.x_opt, self.v_opt, sum_weight = x_opt1, v_opt1, sum_weight1

        else:
            self.x_opt, self.v_opt, sum_weight = x_opt2, v_opt2, sum_weight2

        return self.x_opt, self.v_opt, sum_weight


class KnapsackBranchBoundSolover(KnapsackSolover):
    def solve(self) -> Tuple[array, float, float]:
        self.v_opt = 0
        self.x_opt = np.zeros(self.N)

        self.BranchBound(self.N, 0, np.zeros(self.N), self.capacity,
                         0, 0, self.values, self.weights)

        sum_weight = np.sum(self.x_opt * self.weights)

        return self.x_opt, self.v_opt, sum_weight

    def BranchBound(self, N: int, i: int, x: array, capacity: float,
                    tmp_v: float, tmp_w: float,
                    values: array, weights: array) -> None:
        if i >= N:
            if tmp_v > self.v_opt:
                self.v_opt = tmp_v
                self.x_opt = x.copy()
        else:
            v_i = values[i]
            last_v = values[i+1:].sum()
            w_i = weights[i]
            if tmp_w + w_i < capacity and tmp_v + v_i + last_v > self.v_opt:
                x[i] = 1
                self.BranchBound(N, i + 1, x, capacity, tmp_v + v_i,
                                 tmp_w + w_i, values, weights)

            if tmp_v + last_v > self.v_opt:
                x[i] = 0
                self.BranchBound(N, i + 1, x, capacity, tmp_v,
                                 tmp_w, values, weights)


if __name__ == "__main__":
    print("N = ", end="")

    N = int(input())
    capacity, values, weights = create_knapsack_instance(N)

    print(f"capacity = {capacity}")
    print(f"values = {values}")
    print(f"weight = {weights}")
    print(f"ratio = {values / weights}")

    print("Greedy")
    KGS = KnapsackGreedySolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KGS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")

    print("HalfApprox")
    KHAS = KnapsackHalfAproxSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KHAS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")

    print("BranchBound")
    KBBS = KnapsackBranchBoundSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KBBS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")
