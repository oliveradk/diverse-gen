from enum import Enum 

class LossType(Enum):
    FOCAL = 'focal'
    EXP = 'exp'
    PROB = 'prob'
    CONF = 'conf'
    DIVDIS = 'divdis'
    DBAT = 'dbat'