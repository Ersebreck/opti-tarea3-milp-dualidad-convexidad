"""Ej. 2a: branch-and-bound para la mochila 0-1 (relajación resuelta con linprog)."""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

v = np.array([2, 3, 4, 6, 8])        # volúmenes
c = np.array([10, 14, 31, 48, 60])   # ingresos
CAP = 11


def relajacion(lb, ub):
    res = linprog(-c, A_ub=[v], b_ub=[CAP], bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        return None, None
    return -res.fun, res.x


incumbente = {"z": -np.inf, "x": None}
nodos = []


def bb(lb, ub, nombre, ramas):
    z, x = relajacion(lb, ub)
    if z is None:
        nodos.append((nombre, ramas, "infactible", None, None))
        return
    frac = [i for i in range(5) if abs(x[i] - round(x[i])) > 1e-9]
    xs = [str(F(xi).limit_denominator(100)) for xi in x]
    if z <= incumbente["z"] + 1e-9:
        nodos.append((nombre, ramas, "podado por cota", z, xs))
        return
    if not frac:
        incumbente.update(z=z, x=x.round().astype(int))
        nodos.append((nombre, ramas, "entero -> nueva incumbente", z, xs))
        return
    i = frac[0]
    nodos.append((nombre, ramas, f"ramificar en x{i+1}", z, xs))
    # primero la rama x_i = 0 (búsqueda en profundidad)
    for val, suf in [(0, "a"), (1, "b")]:
        lb2, ub2 = lb.copy(), ub.copy()
        lb2[i] = ub2[i] = val
        bb(lb2, ub2, nombre + suf, ramas + [f"x{i+1}={val}"])


bb([0] * 5, [1] * 5, "P0", [])
for n in nodos:
    print(n)
print("óptimo:", incumbente)
