"""Ej. 2b-c: dual de la relajación lineal y dual entero de la mochila 0-1."""
import itertools
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

v = np.array([2, 3, 4, 6, 8])
c = np.array([10, 14, 31, 48, 60])
CAP = 11
A = np.vstack([v, np.eye(5)])          # capacidad + cotas x_i <= 1
b = np.r_[CAP, np.ones(5)]

# (b) relajación lineal (P) y su dual (D)
p = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * 5, method="highs")
d = linprog(b, A_ub=-A.T, b_ub=-c, bounds=[(0, None)] * 6, method="highs")
print("(P) x =", p.x, " z =", -p.fun)
print("(D) (y, u) =", d.x, " w =", d.fun)
print("    multiplicadores de (P):", -p.ineqlin.marginals)

# óptimo entero primal por enumeración (verificación de 2a)
mejor = max((c @ np.array(s), s) for s in itertools.product([0, 1], repeat=5)
            if v @ np.array(s) <= CAP)
x_ent = np.array(mejor[1])
print("\nPrimal entero: x* =", x_ent, " z* =", mejor[0])

# (c) dual entero
de = milp(b, constraints=LinearConstraint(A.T, lb=c), integrality=np.ones(6),
          bounds=Bounds(0, np.inf))
lam = np.round(de.x).astype(int)
print("Dual entero: (y, u) =", lam, " w* =", de.fun)
print("g(y) =", {y: int(CAP * y + np.maximum(0, c - v * y).sum()) for y in range(5, 11)})

# holgura complementaria en los óptimos enteros
y, u = lam[0], lam[1:]
hc1 = x_ent * (v * y + u - c)
hc2 = y * (CAP - v @ x_ent)
hc3 = u * (1 - x_ent)
print("\nHC1  x_i (v_i y + u_i - c_i) =", hc1)
print("HC2  y (11 - v^T x)          =", hc2)
print("HC3  u_i (1 - x_i)           =", hc3)
print("Suma =", hc1.sum() + hc2 + hc3.sum(), " brecha w* - z* =", de.fun - mejor[0])
