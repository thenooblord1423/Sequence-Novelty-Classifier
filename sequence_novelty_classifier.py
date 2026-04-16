import math
import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
import warnings

class SequenceNoveltyClassifier:
    def __init__(self, sequence):
        self.raw_sequence = sequence
        # Ensure self.seq starts as a list to avoid tuple errors
        self.seq = list(sequence)
        
    def step1_normalize(self):
        seq = list(self.raw_sequence)
        
        while len(seq) > 0 and seq[0] == 0:
            seq = seq[1:]
            
        if len(seq) < 5:
            self.seq = seq
            return
            
        gcd_val = seq[0]
        for val in seq[1:]:
            gcd_val = math.gcd(gcd_val, val)
        if gcd_val > 1:
            seq = [x // gcd_val for x in seq]
            
        best_seq = seq
        min_entropy = float('inf')
        
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
        s = self.seq
        n = len(s)
        b = [1] + [0] * (n - 1)
        c = [1] + [0] * (n - 1)
        L = 0
        m = -1
        
        for i in range(n):
            d = s[i] + sum(c[j] * s[i - j] for j in range(1, L + 1))
            if d != 0:
                temp = c[:] # Reliable slice copy
                for j in range(n - i + m):
                    if i - m + j < n:
                        c[i - m + j] = c[i - m + j] - d * b[j]
                if 2 * L <= i:
                    L = i + 1 - L
                    m = i
                    b = [x / d for x in temp]
        
        return L > 0 and L <= n // 2

    def step4_growth_rate(self):
        if len(self.seq) < 10 or any(x <= 0 for x in self.seq):
            return False
            
        y_data = np.log(self.seq)
        x_data = np.arange(1, len(self.seq) + 1)
        
        def growth_model(n, log_lambda, alpha, C):
            return n * log_lambda + alpha * np.log(n) + C
            
        try:
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(growth_model, x_data, y_data, p0=[1.0, 0.0, 1.0])
            residuals = y_data - growth_model(x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return r_squared > 0.999
        except:
            return False

    def step5_structural_flags(self):
        zero_count = sum(1 for x in self.seq if x == 0)
        if zero_count > len(self.seq) * 0.3:
            return True
            
        if len(self.seq) > 3:
            ratios = [self.seq[i]/self.seq[i-1] for i in range(1, len(self.seq)) if self.seq[i-1] != 0]
            if len(ratios) > 1:
                return np.std(ratios) > np.mean(ratios)
        return False

    def classify(self):
        self.step1_normalize()
        if not self.seq: return "YES"
        if self.step2_binomial_structure(): return "NO"
        if self.step3_linear_recurrence(): return "NO"
        if self.step4_growth_rate(): return "NO"
        return "YES" if self.step5_structural_flags() else "YES"

# === Example Usage ===
if __name__ == "__main__":
    fib = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55) # Fibonacci Sequence
    clf_fib = SequenceNoveltyClassifier(fib)
    print(f"Fibonacci Classification: {clf_fib.classify()}") 
    
    novel_tiling = (1, 0, 0, 2, 0, 0, 0, 0, 15) # Sequence 21 in Appendix A
    clf_novel = SequenceNoveltyClassifier(novel_tiling)
    print(f"Novel Tiling Classification: {clf_novel.classify()}")