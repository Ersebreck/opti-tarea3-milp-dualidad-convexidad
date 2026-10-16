"""Ej. 1: verificación del modelo MILP con big-M (HiGHS vía scipy.optimize.milp)."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

EPS = 1e-4

# variables: (x1, x2, delta)
c = -np.array([2, 4, 3])                       # milp minimiza: se usa -objetivo
A = np.array([[1, 2, 0],                       # x1 + 2 x2 <= 4
              [1, -1, -1],                     # (i)  x1 - x2 - delta >= -2
              [1, -1, -(5 + EPS)]])            # (ii) x1 - x2 - (5+eps) delta <= -1 - eps
restr = LinearConstraint(A, lb=[-np.inf, -2, -np.inf], ub=[4, np.inf, -1 - EPS])
integralidad = np.array([0, 0, 1])             # delta entera (binaria por sus cotas)


def resolver(delta_fijo=None):
    lb, ub = [0, 0, 0], [np.inf, np.inf, 1]
    if delta_fijo is not None:
        lb[2] = ub[2] = delta_fijo
    res = milp(c, constraints=restr, integrality=integralidad, bounds=Bounds(lb, ub))
    return -res.fun, *res.x


for caso in (None, 0, 1):
    z, x1, x2, d = resolver(caso)
    etiqueta = "libre" if caso is None else f"fijo en {caso}"
    print(f"delta {etiqueta:9s}: z = {z:6.2f}, (x1, x2, delta) = ({x1:.4f}, {x2:.4f}, {d:.0f})")
