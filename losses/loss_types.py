from enum import Enum 

class LossType(Enum):
    # FOCAL = 'focal'
    EXP = 'exp'
    PROB = 'prob'
    TOPK = 'topk'
    CONF = 'conf'
    DIVDIS = 'divdis'
    DBAT = 'dbat'