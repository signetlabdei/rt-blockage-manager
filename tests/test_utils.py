# AUTHOR(S):
# Mattia Lecci <mattia.lecci@dei.unipd.it>
# 
# University of Padova (UNIPD), Italy
# Information Engineering Department (DEI) 
# SIGNET Research Group @ http://signet.dei.unipd.it/
# 
# Date: January 2021

from src.utils import *
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
