import time
from typing import Tuple

import numpy as np

from knapsack import (
    KnapsackSolover,
    KnapsackGreedySolover,
    create_knapsack_instance
)

from knapsack3 import KnapsackLinearBranchBoundSolover

array = np.ndarray


class KnapsackPeggedLinearBranchBoundSolover(KnapsackSolover):
    """ 線形緩和問題を用いた分枝限定法でナップザック問題を解くクラス """

    def solve(self) -> Tuple[array, float, float]:

        KGS = KnapsackGreedySolover(
            self.capacity, self.values, self.weights
        )
        self.x_opt, self.v_opt, _ = KGS.solve()

        test_res = np.array([self.pegging_test(idx) for idx in range(self.N)])

        fixed_indices = np.arange(self.N)[test_res[:, 1] == 1].tolist()
        x_tmp = test_res[:, 0]

        self.BranchBound(x_tmp, fixed_indices)

        sum_weight = np.sum(self.x_opt * self.weights)

        return self.x_opt, self.v_opt, sum_weight

    def pegging_test(self, idx: int) -> list:
        x0 = np.zeros(self.N)
        x1 = x0.copy()
        x1[idx] = 1

        _, v0, _ = self.relaxed_linear(x0, [idx])
        _, v1, _ = self.relaxed_linear(x1, [idx])

        if v0 < self.v_opt:
            return [1, 1]

        elif v1 < self.v_opt:
            return [0, 1]

        else:
            return [0, 0]

    def BranchBound(self, x_tmp: array, fixed_indices: list) -> None:
        """ 深さ優先探索で分枝限定を行う再帰関数 """
        x_lin, v_lin, float_index = self.relaxed_linear(
            x_tmp, fixed_indices
        )

        if v_lin < self.v_opt:  # 線形緩和解が暫定解より小さい
            return None

        elif float_index == -1:  # 線形緩和解が暫定解以上かつ実行可能
            self.v_opt = v_lin
            self.x_opt = x_lin.copy()
            return None

        else:  # 線形緩和解が暫定解以上かつ実行不可能
            x = x_tmp.copy()
            tmp_v = (x * self.values).sum()
            tmp_w = (x * self.weights).sum()

            next_fixed_indices = fixed_indices + [float_index]
            v_i = self.values[float_index]
            last_v = self.values.sum() - self.values[next_fixed_indices].sum()
            w_i = self.weights[float_index]
            if tmp_w + w_i < self.capacity and \
                    tmp_v + v_i + last_v > self.v_opt:
                x[float_index] = 1
                self.BranchBound(x, next_fixed_indices)

            if tmp_v + last_v > self.v_opt:
                x[float_index] = 0
                self.BranchBound(x, next_fixed_indices)

            return

    def relaxed_linear(self, x_tmp: array,
                       fixed_indices: list) -> [array, float, int]:
        """ 線形緩和問題を解くメソッド """

        x = x_tmp.copy()
        sum_value = self.values[x_tmp == 1].sum()
        sum_weight = self.weights[x_tmp == 1].sum()

        float_index = -1
        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            if i not in fixed_indices:
                if sum_weight + w > self.capacity:
                    ratio = (self.capacity - sum_weight) / w
                    sum_weight = int(sum_weight + w * ratio)
                    sum_value = int(sum_value + v * ratio)
                    x[i] = ratio
                    float_index = i
                    break
                else:
                    sum_weight += w
                    sum_value += v
                    x[i] = 1

        return x, sum_value, float_index


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

    print("LinearBranchBound")
    KHAS = KnapsackLinearBranchBoundSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KHAS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")

    print("LinearBranchBound")
    KPLBBS = KnapsackPeggedLinearBranchBoundSolover(capacity, values, weights)
    s = time.time()
    x_opt, v_opt, w_opt = KPLBBS.solve()
    e = time.time()
    print(f"x_opt = {x_opt}, v_opt = {v_opt}, w_opt = {w_opt}, time = {e - s}")
