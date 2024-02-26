import numpy as np
from rbssim2.fortran import Stopping as fStopping
from rbssim2 import _Stopping
from rbssim2.Globals import STOPPING_FOLDER


params = np.loadtxt(f'{STOPPING_FOLDER}2_4_6.dat',)
energy = np.arange(3000, 6000, 1., )

def test_inverse():

    np.testing.assert_allclose(
        _Stopping.inverse(energy, energy.size, params, params.size),
        fStopping.inverse(energy, energy.size, params, params.size), atol=0.01
    )

def test_equation():

    np.testing.assert_allclose(
        _Stopping.equation(energy, energy.size, params, params.size),
        fStopping.equation(energy, energy.size, params, params.size), atol=0.01
    )

def test_inveseIntergal():

    np.testing.assert_allclose(
        _Stopping.inverseintegral(energy, energy.size, params, params.size),
        fStopping.inverseintegral(energy, energy.size, params, params.size), atol=0.01
    )

def test_inverseIntegrate():

    np.testing.assert_allclose(
        _Stopping.inverseintegrate(energy, energy, energy.size, params, params.size),
        fStopping.inverseintegrate(energy, energy, energy.size, params, params.size), atol=0.01
    )

def test_inverseDiff():

    np.testing.assert_allclose(
        _Stopping.inversediff(energy, energy.size, params, params.size),
        fStopping.inversediff(energy, energy.size, params, params.size), atol=0.01
    )

def test_EnergyAfterStopping():

    X = _Stopping.inverse(energy, energy.size, params, params.size)
    E = np.ones_like(X) * 6000
    np.testing.assert_allclose(
        _Stopping.energyafterstopping(E, X, energy.size, params, params.size, 200),
        fStopping.energyafterstopping(E, X, energy.size, params, params.size, 200), atol=0.01
    )
