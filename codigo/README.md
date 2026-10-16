# Tarea 3 – Código de verificación

Scripts que verifican numérica o simbólicamente los resultados del informe.

| Script | Ejercicio | Qué verifica |
|---|---|---|
| `ej1_bigM.py` | 1 | Modelo MILP con big-M (HiGHS vía `scipy.optimize.milp`): óptimo z* = 11 y caso δ = 0 |
| `ej2a_bb.py` | 2a | Árbol de Branch-and-Bound completo (15 nodos), óptimo z* = 79 |
| `ej2bc_dual.py` | 2b–c | Dual de la relajación, dual entero y violación de la holgura complementaria |
| `ej3_qp_caja.py` | 3 | P ≻ 0, gradiente en x* y solución numérica con cotas (L-BFGS-B) |
| `ej4_sympy.py` | 4 | Punto crítico, Hessiana, menores y valor mínimo (cálculo simbólico) |

## Uso

```bash
python -m venv .venv
source .venv/bin/activate        # en fish: source .venv/bin/activate.fish
pip install -r requirements.txt
python ej1_bigM.py
```

Requiere Python ≥ 3.9 con `numpy`, `scipy` (≥ 1.9, por `milp`) y `sympy`. No usa PuLP.
