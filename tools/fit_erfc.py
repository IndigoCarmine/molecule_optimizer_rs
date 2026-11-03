"""Fits the Chebyshev series used by src/fast/approx/erfc.rs.

h(x) = (x + K) * exp(x^2) * erfc(x) is fitted in t = (x - K)/(x + K) on
x in [0, X] against 110-digit decimal reference values.

Usage: python tools/fit_erfc.py X K DEGREE [print]
(the monomial coefficients in erfc.rs come from
`python tools/fit_erfc.py 8 2.0 20 print`).
"""
from decimal import Decimal as D, getcontext
import numpy as np, sys
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


def erfcx_hp(x):
    return erfc_hp(x) * (D(x) * D(x)).exp()


X = float(sys.argv[1])
K = float(sys.argv[2])
deg = int(sys.argv[3])
xs = np.linspace(0, X, 800)
xs = np.concatenate([xs, X * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2])
hp = [erfcx_hp(float(x)) for x in xs]
t = (xs - K) / (xs + K)
t0 = -1.0
t1 = (X - K) / (X + K)
h = np.array([float(v * (D(K) + D(float(x)))) for v, x in zip(hp, xs)])
# fit monomial in t directly via chebyshev then convert
u = (2 * t - (t0 + t1)) / (t1 - t0)
V = np.polynomial.chebyshev.chebvander(u, deg)
c, *_ = np.linalg.lstsq(V, h, rcond=None)
approx = np.polynomial.chebyshev.chebval(u, c) / (K + xs)
err = max(abs(float((D(float(a)) - v) / v)) for a, v in zip(approx, hp))
print(deg, K, err)
if len(sys.argv) > 4:
    # erfc.rs evaluates the series as a polynomial in u (monomial basis)
    print("t0", t0, "t1", t1, "monomial coefficients in u:")
    for v in np.polynomial.chebyshev.cheb2poly(c):
        print(repr(float(v)))
