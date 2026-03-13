#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

POINTS: Dict[str, Tuple[float, float, float, float]] = {
    "A": (0, 0, 1, 1),
    "C": (0, 0, 0, 1),
    "E": (0, 0, 1, 0),
    "B": (1, 0, 1, 1),
    "D": (1, 0, 1, 0),
    "F": (1, 0, 0, 1),
}

GT: Dict[str, int] = {k: 1 for k in ["A", "C", "E"]}
GT.update({k: 2 for k in ["B", "D", "F"]})


def threshold_class(score: float, reverted: bool) -> int:
    # normal: score>=0 -> class1
    # reverted: score>=0 -> class2
    return 1 if (score >= 0) ^ reverted else 2


def misclassified_count(w1, w2, w3, w4, w0, reverted: bool) -> int:
    cnt = 0
    for name, (x1, x2, x3, x4) in POINTS.items():
        score = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w0
        pred = threshold_class(score, reverted)
        if pred != GT[name]:
            cnt += 1
    return cnt


def parse_floats(line: str, n: int) -> Tuple[float, ...]:
    parts = line.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"Need exactly {n} numbers, got {len(parts)}.")
    return tuple(float(p) for p in parts)


def is_optimal(w1, w2, w3, w4, w0, tol: float = 1e-9) -> bool:
    def close(a, b): return abs(a - b) <= tol
    return close(w1, -2.0*w0)  and close(w3, 0.0) and close(w4, 0.0)


def main():
    print("Input: w1 w2 w3 w4 w0")
    print("Output: feasible, optimal")
    print("Type 'q' to quit.\n")

    while True:
        try:
            line = input("w1 w2 w3 w4 w0: ").strip()
            if line.lower() in {"q", "quit", "exit"}:
                break

            w1, w2, w3, w4, w0 = parse_floats(line, 5)

            mis_norm = misclassified_count(w1, w2, w3, w4, w0, reverted=False)
            mis_rev  = misclassified_count(w1, w2, w3, w4, w0, reverted=True)

            feasible = (mis_norm == 0) or (mis_rev == 0)
            optimal = is_optimal(w1, w2, w3, w4, w0)

            print(f"feasible={feasible}  optimal={optimal}\n")

        except Exception as e:
            print(f"[Error] {e}\n")


if __name__ == "__main__":
    main()
