import numpy as np
from rbssim2.fortran import utilsrbs as futilsRBS
from rbssim2 import utilsRBS
from rbssim2.Globals import STOPPING_FOLDER

E = np.arange(1000, 2000, 2.)
a = 1.
b = 1500.
c = 20.
d = 0.

def test_gauss():

    np.testing.assert_allclose(
        utilsRBS.gauss(E, a, b, c, d),
        futilsRBS.gauss(E, len(E), a, b, c, d), rtol=1e-2
    )


def test_get_spread_responce():

    spread = np.ones_like(E) * 30.
    k = 0.25

    np.testing.assert_allclose(
        utilsRBS.get_spread_responce(E, spread, E.size, k),
        futilsRBS.get_spread_responce(E, spread, E.size, k), atol=1e-2
    )