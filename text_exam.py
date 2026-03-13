#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

# ================= Dataset (4D) =================
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


# ================= Classifier =================

def threshold_class(z: float, reverted: bool) -> int:
    return 1 if (z >= 0) ^ reverted else 2
    # XOR trick:
    # reverted=False → normal
    # reverted=True  → flipped


def predict(point, w1, w2, w3, w4, w0, reverted):
    x1, x2, x3, x4 = point
    score = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w0
    return threshold_class(score, reverted)


def misclassified_points(w1, w2, w3, w4, w0, reverted, N=(0,0,0,0)) -> List[str]:
    nx1, nx2, nx3, nx4 = N
    wrong = []

    for name, (x1, x2, x3, x4) in POINTS.items():
        shifted = (x1+nx1, x2+nx2, x3+nx3, x4+nx4)
        pred = predict(shifted, w1, w2, w3, w4, w0, reverted)
        if pred != GT[name]:
            wrong.append(name)

    return wrong


def parse_floats(line: str, n: int):
    parts = line.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"Need {n} numbers.")
    return tuple(float(p) for p in parts)


# ================= Main Loop =================

def main():
    print("4D Linear Classifier Explorer")
    print("z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w0")
    print("Both NORMAL and REVERTED modes will be shown.\n")

    while True:
        try:
            line = input("Enter w1 w2 w3 w4 w0: ").strip()
            if line.lower() in {"q", "quit"}:
                break
            w1, w2, w3, w4, w0 = parse_floats(line, 5)

            line = input("Enter N vector (nx1 nx2 nx3 nx4), default 0 0 0 0: ").strip()
            N = (0,0,0,0) if line == "" else parse_floats(line, 4)

            # Normal
            wrong_before_norm = misclassified_points(w1,w2,w3,w4,w0,False,(0,0,0,0))
            wrong_after_norm  = misclassified_points(w1,w2,w3,w4,w0,False,N)

            # Reverted
            wrong_before_rev = misclassified_points(w1,w2,w3,w4,w0,True,(0,0,0,0))
            wrong_after_rev  = misclassified_points(w1,w2,w3,w4,w0,True,N)

            print("\n=== NORMAL MODE (z>=0 → class1) ===")
            print("Misclassified BEFORE N:", wrong_before_norm)
            print("Misclassified AFTER  N:", wrong_after_norm)

            print("\n=== REVERTED MODE (z>=0 → class2) ===")
            print("Misclassified BEFORE N:", wrong_before_rev)
            print("Misclassified AFTER  N:", wrong_after_rev)
            print()

        except Exception as e:
            print("[Error]", e)


if __name__ == "__main__":
    main()
