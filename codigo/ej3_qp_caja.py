"""Ej. 3: verificación de la optimalidad de x* en el QP con restricciones de caja."""
import numpy as np
from scipy.optimize import minimize

P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]], dtype=float)
q = np.array([-22.0, -14.5, 13.0])
r = 1.0
x_star = np.array([1.0, 0.5, -1.0])


def f(x):
    return 0.5 * x @ P @ x + q @ x + r


print("Menores principales:", [round(float(np.linalg.det(P[:k, :k])), 6) for k in (1, 2, 3)])
print("Valores propios de P:", np.linalg.eigvalsh(P))
print("Gradiente en x*:", P @ x_star + q)
print("f(x*) =", f(x_star))
print("Mínimo sin restricciones -P^{-1} q =", np.linalg.solve(P, -q))

res = minimize(f, np.zeros(3), jac=lambda x: P @ x + q,
               bounds=[(-1, 1)] * 3, method="L-BFGS-B")
print("L-BFGS-B con cotas: x =", res.x, " f =", res.fun)
