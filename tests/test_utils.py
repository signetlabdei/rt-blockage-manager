# AUTHOR(S):
# Paolo Testolina <paolo.testolina@dei.unipd.it>
# Alessandro Traspadini <alessandro.traspadini@dei.unipd.it>
# Mattia Lecci <mattia.lecci@dei.unipd.it>
#
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI)
# SIGNET Research Group @ http://signet.dei.unipd.it/
#
# Date: January 2021

from src.utils import get_wavelength, isint, isfloat
import pytest


@pytest.mark.parametrize("x,answer", [('1', True),
                                      ('-1', True),
                                      ('1.0', False),
                                      ('-1.0', False),
                                      ('1e-2', False),
                                      ('1.1125', False),
                                      ('-1.3e-5', False),
                                      ('inf', False),
                                      ('nan', False),
                                      ('a', False),
                                      ('0xff', False),
                                      ('0b10', False)])
def test_isint(x, answer):
    assert isint(x) == answer


@pytest.mark.parametrize("x,answer", [('1', True),
                                      ('-1', True),
                                      ('1.0', True),
                                      ('-1.0', True),
                                      ('1e-2', True),
                                      ('1.1125', True),
                                      ('-1.3e-5', True),
                                      ('inf', True),
                                      ('nan', True),
                                      ('a', False),
                                      ('0xff', False),
                                      ('0b10', False)])
def test_isfloat(x, answer):
    assert isfloat(x) == answer


@pytest.mark.parametrize("freq,wavelength", [(1e6, 300),
                                             (10e6, 30),
                                             (100e6, 3),
                                             (1e9, 0.3),
                                             (1.2e9, 0.25),
                                             (10e9, 0.03)])
def test_get_wavelength(freq, wavelength):
    assert get_wavelength(freq) == pytest.approx(wavelength, rel=1e-3)
