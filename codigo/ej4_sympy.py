"""Ej. 4: condiciones de primer y segundo orden calculadas simbólicamente."""
import sympy as sp

x, y, z = sp.symbols("x y z")
f = 2*x**2 + x*y + y**2 + y*z + z**2 - 6*x - 7*y - 8*z + 9

grad = [sp.diff(f, v) for v in (x, y, z)]
punto = sp.solve(grad, (x, y, z), dict=True)[0]
H = sp.hessian(f, (x, y, z))

print("Gradiente:", grad)
print("Punto crítico:", punto)
print("Hessiana:", H.tolist())
print("Menores principales:", [H[:k, :k].det() for k in (1, 2, 3)])
print("Polinomio característico:", H.charpoly().as_expr())
print("Valores propios:", sorted(sp.re(sp.N(e, 4)) for e in H.eigenvals()))
print("f(x*) =", f.subs(punto))
