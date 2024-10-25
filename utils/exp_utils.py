
OUT_DIR = "output"

def get_cifar_mnist_exp_dir(dict_args: dict):
    dir_name = f"{OUT_DIR}/cifar_mnist/{dict_args['loss_type'].name}_{dict_args['model']}_{dict_args['mix_rate']}_{dict_args['mix_rate_lower_bound']}_{dict_args['lr']}_{dict_args['aux_weight']}_{dict_args['epochs']}_{dict_args['seed']}"
    return dir_name
