import numpy as np

# A log-verossimilhanca e sempre superior a funcao Q
# calculada no passo E.
# L(theta) >= Q(theta, theta_velho) + Constante
# Ao maximizar Q, garantimos que L nao diminui.