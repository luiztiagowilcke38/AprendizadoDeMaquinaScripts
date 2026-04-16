import numpy as np

# A matriz Omega que penaliza a curvatura no ESL:
# Omega_jk = integral N''_j(x) N''_k(x) dx
# Esta matriz e esparsa (bandada) para B-splines.
# O escalonamento de lambda controla o compromisso Bias-Var.