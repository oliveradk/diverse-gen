from enum import Enum

def conf_to_args(conf: dict):
    args = []
    for key, value in conf.items():
        # check if value is an enum 
        if isinstance(value, Enum):
            value = value.name 
        elif value is None:
            value = 'null'
        args.append(f"{key}={value}")
    return args