import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### tested July 17th 2024. It works!
def hardsphere(q, radius_effective, volfraction):

    X = 1.0 / (1.0 - volfraction)
    D = X * X
    A = ((1. + 2. * volfraction) * D)**2
    X = np.abs(q * radius_effective * 2.0)

    if (X.all() < 5e-06):
        HARDSPH = 1./A

    X2 = X**2
    B = (1.0 + 0.5 * volfraction) * D
    B *= B
    B *= -6. * volfraction
    G = 0.5 * volfraction * A

    CUTOFFHS = 0.4
    if X.all() < CUTOFFHS:
        FF = 8.0 * A + 6.0 * B + 4.0 * G + (-0.8 * A - B / 1.5 - 0.5 * G + (A / 35. + 0.0125 * B + 0.02 * G) * X2) * X2
        HARDSPH = 1./(1. + volfrac * FF)

    X4 = X2**2
    S = np.sin(X)
    C = np.cos(X)
    FF = ((G * ((4. * X2 - 24.) * X * S - (X4 - 12. * X2 + 24.) * C + 24.) / X2 + B * (
                2. * X * S - (X2 - 2.) * C - 2.)) / X + A * (S - X * C)) / X
    HARDSPH = 1. / (1. + 24. * volfraction * FF / X2)

    return HARDSPH
