"""
Sequence Novelty Classifier

Implementation of Algorithm 1 (Section 5.2) of
"Structural equivalence and novelty in constrained tiling sequences."

    Input : sequence prefix a_0, ..., a_N  (15 to 30 terms)
    Output: "NO"  -> reducible to a known structural class
            "YES" -> no reduction detected (candidate novel, not a proof of novelty)

Each step below corresponds to the step of the same number in the paper.
Sequence terms are kept as exact Python integers, so large terms do not overflow.
"""

import math
from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np


def binom(n, k):
    """Binomial coefficient, defined as 0 outside 0 <= k <= n."""
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


class SequenceNoveltyClassifier:
    """
    Parameters
    ----------
    sequence : list[int]
        Prefix a_0, ..., a_N of the sequence (15 to 30 terms).
    rule_features : set[str], optional
        Features of the sequence's generative rule, examined in Step 5:
        "nonlinear", "arithmetic", "nonlocal". A list of numbers cannot
        reveal how it was generated, so these are supplied with the sequence.
    """

    SHIFT_RANGE = 5        # Step 1: index shifts n -> n + k, k in [-5, 5]
    MAX_ORDER = 3          # Step 3: bounded recurrence order d
    MAX_DEGREE = 2         # Step 3: bounded degree of the coefficient polynomials
    R2_SMOOTH = 0.9999     # Step 4: threshold for a smooth exponential fit

    def __init__(self, sequence, rule_features=None):
        self.raw = [int(x) for x in sequence]
        self.rule_features = set(rule_features or [])
        self.trace = []

    # ------------------------------------------------------------------
    # Step 1: Normalization
    # ------------------------------------------------------------------
    def step1_normalize(self):
        """Strip leading zeros, divide out gcd(a_0, ..., a_N), test index
        shifts n -> n + k for k in [-5, 5], and select the normalization that
        minimizes the entropy of the finite differences."""
        g = 0
        for x in self.raw:
            g = math.gcd(g, abs(x))
        g = g or 1
        self.scaled = [x // g for x in self.raw]          # gcd removed, original indexing

        seq = list(self.scaled)
        lead = 0
        while seq and seq[0] == 0:                        # strip leading zeros
            seq.pop(0)
            lead += 1

        def entropy(s):
            d = [b - a for a, b in zip(s, s[1:])]
            if not d:
                return 0.0
            c = Counter(d)
            return -sum((v / len(d)) * math.log2(v / len(d)) for v in c.values())

        best_k, best_seq, best_h = 0, seq, entropy(seq)
        for k in range(-self.SHIFT_RANGE, self.SHIFT_RANGE + 1):
            # k > 0 drops the first k terms; k <= 0 only relabels the index,
            # which leaves the finite differences (and their entropy) unchanged.
            s = seq[k:] if k > 0 else seq
            if len(s) < 8:
                continue
            h = entropy(s)
            if h < best_h - 1e-12:
                best_k, best_seq, best_h = k, s, h
        self.seq = best_seq
        self.trace.append(f"Step 1: stripped {lead} leading zero(s), gcd = {g}, shift k = {best_k}")

    # ------------------------------------------------------------------
    # Step 2: Detection of closed-form binomial structure
    # ------------------------------------------------------------------
    # Candidate binomial families a_n = sum_k f(n,k) g(n,k), where
    #   f(n,k) = C(A1*n + B1*k + C1, D1*k + E1)
    #   g(n,k) = C(A2*n + B2*k + C2, D2*k + E2)   (or g = 1)
    # with small integer parameters.
    _GRID = list(product((0, 1, 2), (-3, -2, -1, 0, 1, 2), (-1, 0, 1), (1, 2), (-1, 0)))

    @staticmethod
    def _family_value(f, g, n):
        a1, b1, c1, d1, e1 = f
        total = 0
        for k in range(n + 2):
            t = binom(a1 * n + b1 * k + c1, d1 * k + e1)
            if t and g is not None:
                a2, b2, c2, d2, e2 = g
                t *= binom(a2 * n + b2 * k + c2, d2 * k + e2)
            total += t
        return total

    @staticmethod
    def _log_least_squares(target, values):
        """Least-squares error between log|target| and log|values|."""
        pts = [(math.log(abs(a)), math.log(abs(b))) for a, b in zip(target, values) if a and b]
        if len(pts) < 3:
            return float("inf")
        return sum((x - y) ** 2 for x, y in pts) / len(pts)

    def step2_binomial_structure(self):
        """Fit candidate binomial families on a logarithmic scale (least squares),
        under the index shifts of Step 1, then verify exact agreement for all
        n <= N. An exact match returns True (-> NO)."""
        target = self.scaled
        N = len(target)
        if N < 8:
            return False
        shifts = range(-self.SHIFT_RANGE, self.SHIFT_RANGE + 1)
        head = 6   # candidates must agree exactly on the first terms before fitting

        survivors = []
        for f in self._GRID:
            for g in [None] + self._GRID:
                cache = {}
                def val(n):
                    if n not in cache:
                        cache[n] = self._family_value(f, g, n)
                    return cache[n]
                for t in shifts:
                    idx = [i for i in range(N) if i + t >= 0][:head]
                    if len(idx) == head and all(val(i + t) == target[i] for i in idx):
                        survivors.append((f, g, t))

        def aligned(c):
            f, g, t = c
            idx = [i for i in range(N) if i + t >= 0]
            return [target[i] for i in idx], [self._family_value(f, g, i + t) for i in idx]

        survivors.sort(key=lambda c: self._log_least_squares(*aligned(c)))
        for c in survivors:
            tgt, vals = aligned(c)
            if len(tgt) >= 8 and tgt == vals:                 # exact agreement for all n <= N
                self.trace.append(f"Step 2: exact match with f={c[0]}, g={c[1]}, shift {c[2]}")
                return True
        return False

    # ------------------------------------------------------------------
    # Step 3: Detection of linear recurrence (D-finite test)
    # ------------------------------------------------------------------
    def _check_terms(self):
        """Number of terms reserved for verification beyond those that
        determine a recurrence (guards against coincidental fits)."""
        return max(3, len(self.seq) // 4)

    def _berlekamp_massey(self):
        """Exact Berlekamp-Massey over the rationals; returns (L, C) with
        a_n + C[1] a_{n-1} + ... + C[L] a_{n-L} = 0."""
        s = [Fraction(x) for x in self.seq]
        C, B = [Fraction(1)], [Fraction(1)]
        L, m, b = 0, 1, Fraction(1)
        for n in range(len(s)):
            d = s[n] + sum(C[i] * s[n - i] for i in range(1, L + 1))
            if d == 0:
                m += 1
                continue
            T = C[:]
            C = C + [Fraction(0)] * max(0, len(B) + m - len(C))
            for i, x in enumerate(B):
                C[i + m] -= (d / b) * x
            if 2 * L <= n:
                L, B, b, m = n + 1 - L, T, d, 1
            else:
                m += 1
        return L, C

    def _constant_coefficients(self):
        s = self.seq
        L, C = self._berlekamp_massey()
        if L == 0 or 2 * L + self._check_terms() > len(s):
            return False
        if all(s[n] + sum(C[i] * s[n - i] for i in range(1, L + 1)) == 0 for n in range(L, len(s))):
            self.trace.append(f"Step 3: constant-coefficient recurrence of order {L}")
            return True
        return False

    def _polynomial_coefficients(self):
        """Regression for p_0(n) a_n = sum_{k=1..d} p_k(n) a_{n-k}, with
        polynomials p_k of bounded degree, followed by exact symbolic
        verification of the recovered integer coefficients."""
        s = self.seq
        N = len(s)
        for d in range(1, self.MAX_ORDER + 1):
            for deg in range(0, self.MAX_DEGREE + 1):
                unknowns = (d + 1) * (deg + 1)
                if (N - d) < unknowns + self._check_terms():
                    continue
                # regression: least-squares null vector of the (row-normalized) system
                rows = []
                for n in range(d, N):
                    ints = [n ** j * s[n - i] for i in range(d + 1) for j in range(deg + 1)]
                    big = max(abs(x) for x in ints) or 1          # scale exactly before
                    r = [float(Fraction(x, big)) for x in ints]   # converting (no overflow)
                    nrm = math.sqrt(sum(x * x for x in r)) or 1.0
                    rows.append([x / nrm for x in r])
                _, sv, vt = np.linalg.svd(np.array(rows))
                v = vt[-1]
                if sv[-1] > 1e-8 * sv[0]:
                    continue
                v = v / v[np.argmax(np.abs(v))]
                coef = [Fraction(float(x)).limit_denominator(1000) for x in v]
                if all(c == 0 for c in coef[: deg + 1]):         # p_0 must not vanish
                    continue
                # symbolic (exact) verification on every term
                if all(sum(coef[i * (deg + 1) + j] * n ** j * s[n - i]
                           for i in range(d + 1) for j in range(deg + 1)) == 0
                       for n in range(d, N)):
                    self.trace.append(f"Step 3: polynomial-coefficient recurrence, order {d}, degree {deg}")
                    return True
        return False

    def step3_linear_recurrence(self):
        """Berlekamp-Massey for constant coefficients, then polynomial
        coefficients by regression with symbolic verification."""
        if len(self.seq) < 8:
            return False
        return self._constant_coefficients() or self._polynomial_coefficients()

    # ------------------------------------------------------------------
    # Step 4: Growth rate estimation
    # ------------------------------------------------------------------
    def step4_growth_rate(self):
        """Fit log a_n ~ n log(lambda) + alpha log n (+ constant) by least squares.
        Returns True for a smooth exponential or polynomial-damped exponential fit."""
        pts = [(n, math.log(abs(x))) for n, x in enumerate(self.seq, start=1) if x != 0]
        if len(pts) < 10 or len(pts) < 0.8 * len(self.seq):
            return False                                     # too sparse to call smooth
        n = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        X = np.column_stack([n, np.log(n), np.ones_like(n)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 if ss_tot == 0 else 1 - float(np.sum((y - X @ coef) ** 2)) / ss_tot
        lam = f"{math.exp(coef[0]):.4f}" if coef[0] < 700 else f"e^{coef[0]:.1f}"
        self.trace.append(f"Step 4: lambda = {lam}, alpha = {coef[1]:.3f}, R^2 = {r2:.6f}")
        return r2 > self.R2_SMOOTH

    # ------------------------------------------------------------------
    # Step 5: Structural feature detection
    # ------------------------------------------------------------------
    def step5_structural_flags(self):
        """Return the non-classical features present: nonlinear constraints,
        arithmetic conditions, nonlocal dependencies (from the generative rule),
        and irregular sparsity or discontinuities (from the terms)."""
        flags = [f for f in ("nonlinear", "arithmetic", "nonlocal") if f in self.rule_features]
        s = self.seq
        if s and sum(1 for x in s if x == 0) > 0.3 * len(s):
            flags.append("irregular sparsity")
        # growth ratios |a_i / a_(i-1)|, computed via logs so huge terms cannot overflow
        ratios = [math.exp(min(700.0, math.log(abs(s[i])) - math.log(abs(s[i - 1]))))
                  for i in range(1, len(s)) if s[i - 1] != 0 and s[i] != 0]
        if len(ratios) > 3 and np.mean(ratios) > 0 and np.std(ratios) > 0.5 * np.mean(ratios):
            flags.append("discontinuities")
        return flags

    # ------------------------------------------------------------------
    # Pipeline (Steps 1-6)
    # ------------------------------------------------------------------
    def classify(self, return_reason=False):
        self.trace = []
        self.step1_normalize()
        if not self.seq:
            out = ("YES", "Step 6: default decision (empty after normalization)")
        elif self.step2_binomial_structure():
            out = ("NO", "Step 2: binomial structure")
        elif self.trace.append("Step 2: no exact binomial-sum match") or self.step3_linear_recurrence():
            out = ("NO", "Step 3: linear recurrence")
        elif self.trace.append("Step 3: no verified recurrence found") or self.step4_growth_rate():
            out = ("NO", "Step 4: smooth exponential growth")
        else:
            if not any(t.startswith("Step 4") for t in self.trace):
                self.trace.append("Step 4: too few nonzero terms for a growth fit")
            flags = self.step5_structural_flags()
            if flags:
                out = ("YES", "Step 5: " + ", ".join(flags))
            else:
                out = ("YES", "Step 6: default decision")
        return out if return_reason else out[0]


def parse_terms(text):
    """Turn pasted text like '1, 1, 2, 3, 5' (commas and/or spaces) into integers."""
    cleaned = text.replace(",", " ").replace("\u2026", " ").replace("...", " ")
    return [int(tok) for tok in cleaned.split()]


if __name__ == "__main__":
    import sys

    print("Sequence Novelty Classifier")
    print("Paste the terms of a sequence (separated by commas or spaces), then press Enter.")
    print("Optionally add rule features after a '|', e.g.:  1, 0, 0, 2, 0 | nonlinear")
    print("Recognized features: nonlinear, arithmetic, nonlocal. Press Enter on an empty line to quit.\n")

    lines = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else iter(lambda: input("> "), "")
    for line in lines:
        terms_part, _, feat_part = line.partition("|")
        try:
            terms = parse_terms(terms_part)
        except ValueError:
            print("  Could not read the terms. Use whole numbers separated by commas or spaces.\n")
            continue
        if not terms:
            continue
        feats = {f.strip().lower() for f in feat_part.replace(",", " ").split() if f.strip()}
        unknown = feats - {"nonlinear", "arithmetic", "nonlocal"}
        if unknown:
            print(f"  Ignoring unrecognized feature(s): {', '.join(sorted(unknown))}")
        if len(terms) < 15:
            print(f"  Note: {len(terms)} terms given; the paper uses 15 to 30. "
                  "Steps 2-4 need at least 8 terms and are more reliable with more.")
        clf = SequenceNoveltyClassifier(terms, feats & {"nonlinear", "arithmetic", "nonlocal"})
        label, reason = clf.classify(return_reason=True)
        print(f"  Result: {label}  ({reason})")
        for step in clf.trace:
            print(f"    {step}")
        print()

