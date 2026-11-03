"""Writes tests/data/erfc_reference.rs: (x, erfc(x)) pairs computed with
110-digit decimal arithmetic.

Usage: python tools/gen_erfc_reference.py > tests/data/erfc_reference.rs
"""
from decimal import Decimal as D, getcontext
import random
getcontext().prec = 110
PI = D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196')
SQPI = PI.sqrt()


def erfc_hp(x):
    x = D(x)
    s = D(0)
    term = x
    n = 0
    x2 = x * x
    while True:
        t = term / (2 * n + 1)
        s += t
        if abs(t) < D(10) ** (-100):
            break
        n += 1
        term = -term * x2 / n
    return 1 - 2 * s / SQPI


random.seed(7)
xs = [i * 0.05 for i in range(0, 141)] + [random.uniform(0, 7) for _ in range(60)] + [1e-8, 1e-4, 0.3333333333333333]
print("// (x, erfc(x)) computed with 110-digit decimal arithmetic (erf Taylor series)")
print("const ERFC_REF: &[(f64, f64)] = &[")
for x in xs:
    x = float(x)
    v = erfc_hp(x)
    print(f"    ({x!r}, {float(v)!r}),")
print("];")
