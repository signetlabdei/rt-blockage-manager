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

def isint(x) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False


def isfloat(x) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def get_wavelength(freq_hz: float) -> float:
    assert freq_hz > 0, "Frequency is not positive"
    # speed of light [m/s] over the frequency [Hz]
    return 299792458 / freq_hz
