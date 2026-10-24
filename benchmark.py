"""
30-sequence benchmark for the Sequence Novelty Classifier (Appendix A).

Rows 1-20: known/reducible sequences (ground truth NO).
Rows 21-30: synthetic tiling sequences whose rules fall outside the classes the
            classifier targets (ground truth YES). 

Usage:  python3 benchmark.py [number_of_terms]     (default 20)
"""
import math
import sys
from functools import lru_cache

from sequence_novelty_classifier import SequenceNoveltyClassifier

C = math.comb


def is_prime(m):
    return m > 1 and all(m % d for d in range(2, int(m ** 0.5) + 1))


def is_square(m):
    return m >= 0 and math.isqrt(m) ** 2 == m


FIB_NUMBERS = {1, 2, 3, 5, 8, 13, 21, 34, 55}


def linrec(init, coeffs, N):
    a = list(init)
    while len(a) < N:
        a.append(sum(c * a[-1 - i] for i, c in enumerate(coeffs)))
    return a[:N]


# ---------------------------------------------------------------- synthetic rules
def row21(n):  # number of squares = (number of dominos)^2
    total = 0
    for d in range(n // 2 + 1):
        if d * d + 2 * d == n:
            total += C(d * d + d, d)
    return total


def row22(n):  # number of dominos is prime
    return sum(C(n - k, k) for k in range(n // 2 + 1) if is_prime(k))


def row23(n):  # total number of tiles is prime
    return sum(C(n - k, k) for k in range(n // 2 + 1) if is_prime(n - k))


def row24(n):  # every domino starts on a prime-numbered cell (cells 1..n)
    @lru_cache(None)
    def g(i):
        if i == n + 1:
            return 1
        w = g(i + 1)
        if i + 1 <= n and is_prime(i):
            w += g(i + 2)
        return w
    return g(1)


def row25(n):  # run of squares before each domino strictly longer than the previous run
    @lru_cache(None)
    def g(rem, last):
        t = 1
        for run in range(last + 1, rem - 1):
            t += g(rem - run - 2, run)
        return t
    return g(n, -1)


def row26(n):  # sum of the starting cells of all dominos equals n
    @lru_cache(None)
    def g(i, s):
        if s > n:
            return 0
        if i == n + 1:
            return 1 if s == n else 0
        w = g(i + 1, s)
        if i + 1 <= n:
            w += g(i + 2, s + i)
        return w
    return g(1, 0)


def row27(n):  # number of dominos is a Fibonacci number
    return sum(C(n - k, k) for k in range(n // 2 + 1) if k in FIB_NUMBERS)


def row28(n):  # dominos in left half = (dominos in right half)^2; no domino crosses the midpoint
    L, R = n // 2, n - n // 2
    return sum(C(L - k * k, k * k) * C(R - k, k) for k in range(R // 2 + 1) if 2 * k * k <= L)


def row29(n):  # after the k-th domino, the next run of squares has length >= k
    @lru_cache(None)
    def g(rem, k, need):
        # `need` squares must still be placed before another domino may appear
        if rem == 0:
            return 1
        w = g(rem - 1, k, max(0, need - 1))           # place a square
        if need == 0 and rem >= 2:
            w += g(rem - 2, k + 1, k + 1)             # place the (k+1)-th domino
        return w
    return g(n, 0, 0)


def row30(n):  # recursive constraint count(n) = count(n-1)^2, with count(0) = 2
    return 2 ** (2 ** n)


def benchmark(N):
    fib = linrec([1, 1], [1, 1], N)
    rows = [
        (1, "Fibonacci", fib, None),
        (2, "Catalan", [C(2 * n, n) // (n + 1) for n in range(N)], None),
        (3, "Central binomial", [C(2 * n, n) for n in range(N)], None),
        (4, "Squares", [(n + 1) ** 2 for n in range(N)], None),
        (5, "Factorial", [math.factorial(n + 1) for n in range(N)], None),
        (6, "Powers of 2", [2 ** n for n in range(N)], None),
        (7, "Domino tilings", fib, None),
        (8, "Tribonacci", linrec([1, 1, 2], [1, 1, 1], N), None),
        (9, "Delannoy", [sum(C(n, k) * C(n + k, k) for k in range(n + 1)) for n in range(N)], None),
        (10, "Central trinomial", [sum(C(n, 2 * k) * C(2 * k, k) for k in range(n // 2 + 1)) for n in range(N)], None),
        (11, "Binomial row sum", [sum(C(n, k) for k in range(n + 1)) for n in range(N)], None),
        (12, "Linear recurrence random", [2 ** (n + 1) - 1 for n in range(N)], None),
        (13, "Polynomial n^3", [(n + 1) ** 3 for n in range(N)], None),
        (14, "2^n + n", [2 ** n + n for n in range(N)], None),
        (15, "C(2n,n)/(n+1)", [C(2 * n, n) // (n + 1) for n in range(N)], None),
        (16, "C(n^2, n)", [C(n * n, n) for n in range(N)], None),
        (17, "C(n^2+n, n)", [C(n * n + n, n) for n in range(N)], None),
        (18, "n! 2^n", [math.factorial(n) * 2 ** n for n in range(1, N + 1)], None),
        (19, "Signed factorial", [(-1) ** n * math.factorial(n + 1) for n in range(N)], None),
        (20, "Sum of binomials", [sum(C(n, k) * 2 ** k for k in range(n + 1)) for n in range(N)], None),
        (21, "Squares = dominos^2", [row21(n) for n in range(N)], {"nonlinear"}),
        (22, "# of dominos must be prime", [row22(n) for n in range(N)], {"arithmetic"}),
        (23, "Total # of tiles is prime", [row23(n) for n in range(N)], {"arithmetic"}),
        (24, "Dominos only on prime-numbered cells", [row24(n) for n in range(N)], {"arithmetic"}),
        (25, "Square runs before dominos strictly increase", [row25(n) for n in range(N)], {"nonlocal"}),
        (26, "Domino start positions sum to n", [row26(n) for n in range(N)], {"nonlocal"}),
        (27, "# of dominos is a Fibonacci number", [row27(n) for n in range(N)], {"nonlinear"}),
        (28, "Left-half dominos = (right-half dominos)^2", [row28(n) for n in range(N)], {"nonlocal", "nonlinear"}),
        (29, "Run after k-th domino has length >= k", [row29(n) for n in range(N)], {"nonlocal"}),
        (30, "count(n) = count(n-1)^2", [row30(n) for n in range(N)], {"nonlinear"}),
    ]
    return rows


def short(seq, k=8):
    out = []
    for x in seq[:k]:
        s = str(x)
        out.append(s if len(s) <= 12 else f"~10^{len(s) - 1}")
    return ", ".join(out) + ", ..."


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tp = tn = fp = fn = 0
    print(f"Benchmark with {N} terms per sequence\n")
    print(f"{'#':>2}  {'Truth':5} {'Output':6} {'Decided by':40} Sequence")
    for num, name, seq, feats in benchmark(N):
        truth = "NO" if num <= 20 else "YES"
        out, reason = SequenceNoveltyClassifier(seq, feats).classify(return_reason=True)
        mark = "" if out == truth else "  <-- error"
        if truth == "YES" and out == "YES": tp += 1
        if truth == "NO" and out == "NO": tn += 1
        if truth == "NO" and out == "YES": fp += 1
        if truth == "YES" and out == "NO": fn += 1
        print(f"{num:>2}  {truth:5} {out:6} {reason[:40]:40} {name}: {short(seq)}{mark}")
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"\nTP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Accuracy={acc:.1%}  Precision={prec:.1%}  Recall={rec:.1%}")
