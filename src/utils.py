# AUTHOR(S):
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
