import numpy as np

def scad_threshold(beta, lam, a=3.7):
    """Reflete a regra de limiarizacao do SCAD"""
    abs_beta = np.abs(beta)
    if abs_beta <= 2*lam:
        return np.sign(beta) * np.maximum(0, abs_beta - lam)
    elif abs_beta <= a*lam:
        return ((a-1)*beta - np.sign(beta)*a*lam) / (a-2)
    else:
        return beta # Sem encolhimento para valores muito grandes