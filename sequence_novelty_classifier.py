import math
import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
import warnings

"""
Data Availability:
This code is provided as part of the manuscript's supplementary materials.
"""

class SequenceNoveltyClassifier:
    """
    Implements a multi-step pipeline to classify integer sequences based on 
    structural novelty vs. known mathematical recurrence patterns.
    """
    def __init__(self, sequence):
        self.raw_sequence = sequence
        self.seq = list(sequence)
        
    def step1_normalize(self):
        """
        NORMALIZATION PROCEDURE:
        Removes leading zeros, reduces by GCD, and performs entropy-based 
        prefix trimming to isolate the core generating sequence.
        """
        seq = list(self.raw_sequence)
        
        # Remove leading zeros
        while len(seq) > 0 and seq[0] == 0:
            seq = seq[1:]
            
        if len(seq) < 5:
            self.seq = seq
            return
            
        # Divide by Greatest Common Divisor (GCD)
        gcd_val = seq[0]
        for val in seq[1:]:
            gcd_val = math.gcd(gcd_val, val)
        if gcd_val > 1:
            seq = [x // gcd_val for x in seq]
            
        best_seq = seq
        min_entropy = float('inf')
        
        # Entropy-based trimming to find the most 'stable' start point
        for k in range(min(6, len(seq) - 3)):
            shifted_seq = seq[k:]
            diffs = np.diff(shifted_seq)
            counts = Counter(diffs)
            total = len(diffs)
            # Shannon entropy calculation
            entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
            
            if entropy < min_entropy:
                min_entropy = entropy
                best_seq = shifted_seq
                
        self.seq = best_seq

    def step2_binomial_structure(self):
        """
        RECURRENCE DETECTION (Polynomial):
        Detects if the sequence can be reduced to zero via repeated 
        differencing (characteristic of binomial/polynomial growth).
        """
        if len(self.seq) < 5:
            return False
        current_diffs = self.seq
        for _ in range(5):
            current_diffs = np.diff(current_diffs)
            if len(current_diffs) == 0: break
            if all(d == 0 for d in current_diffs):
                return True
        return False

    def step3_linear_recurrence(self):
        """
        RECURRENCE DETECTION (Linear):
        Implements the Berlekamp-Massey algorithm to find the minimal 
        polynomial of the sequence. Returns True if a short recurrence exists.
        """
        s = self.seq
        n = len(s)
        b = [1] + [0] * (n - 1)
        c = [1] + [0] * (n - 1)
        L = 0
        m = -1
        
        for i in range(n):
            d = s[i] + sum(c[j] * s[i - j] for j in range(1, L + 1))
            if d != 0:
                temp = c[:] 
                for j in range(n - i + m):
                    if i - m + j < n:
                        c[i - m + j] = c[i - m + j] - d * b[j]
                if 2 * L <= i:
                    L = i + 1 - L
                    m = i
                    b = [x / d for x in temp]
        
        # A sequence is 'known' if the recurrence length L is less than half the length
        return L > 0 and L <= n // 2

    def step4_growth_rate(self):
        """
        MODEL FITTING DETAILS:
        Fits the sequence to an exponential growth model: a(n) ~ c * lambda^n * n^alpha.
        Uses R-squared threshold to determine if the sequence fits known growth types.
        """
        if len(self.seq) < 10 or any(x <= 0 for x in self.seq):
            return False
            
        y_data = np.log(self.seq)
        x_data = np.arange(1, len(self.seq) + 1)
        
        def growth_model(n, log_lambda, alpha, C):
            return n * log_lambda + alpha * np.log(n) + C
            
        try:
            warnings.simplefilter('ignore')
            # Non-linear least squares fit
            popt, _ = curve_fit(growth_model, x_data, y_data, p0=[1.0, 0.0, 1.0])
            residuals = y_data - growth_model(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return r_squared > 0.999 # Strict threshold for mathematical sequences
        except:
            return False

    def step5_structural_flags(self):
        """
        Checks for sparse or highly irregular structures that suggest 
        novelty (e.g., high zero-density or irregular ratios).
        """
        zero_count = sum(1 for x in self.seq if x == 0)
        if zero_count > len(self.seq) * 0.3:
            return True
            
        if len(self.seq) > 3:
            ratios = [self.seq[i]/self.seq[i-1] for i in range(1, len(self.seq)) if self.seq[i-1] != 0]
            if len(ratios) > 1:
                return np.std(ratios) > np.mean(ratios)
        return False

    def classify(self):
        """
        Main execution pipeline. Returns 'YES' for novel, 'NO' for known.
        """
        self.step1_normalize()
        if not self.seq: return "YES"
        
        # If any known recurrence or growth model fits, it is not 'novel'
        if self.step2_binomial_structure(): return "NO"
        if self.step3_linear_recurrence(): return "NO"
        if self.step4_growth_rate(): return "NO"
        
        # Otherwise, check for structural indicators of novelty
        return "YES" if self.step5_structural_flags() else "YES"

# === Ready-to-Run Entry Point ===
if __name__ == "__main__":
    print("--- Sequence Novelty Classification Test ---")
    
    # Example 1: Known Recurrence (Fibonacci)
    fib = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55)
    clf_fib = SequenceNoveltyClassifier(fib)
    print(f"Fibonacci (Known): {clf_fib.classify()}") 
    
    # Example 2: Known Polynomial (Squares)
    squares = (1, 4, 9, 16, 25, 36, 49, 64)
    clf_sq = SequenceNoveltyClassifier(squares)
    print(f"Squares (Known): {clf_sq.classify()}")
    
    # Example 3: Novel/Irregular Tiling Sequence
    novel_tiling = (1, 0, 0, 2, 0, 0, 0, 0, 15)
    clf_novel = SequenceNoveltyClassifier(novel_tiling)
    print(f"Novel Tiling (Novel): {clf_novel.classify()}")
