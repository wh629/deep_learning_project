import os
import numpy

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_exp_name(dataset, model, max_length, lr, bs, max_epochs, seed):
    return f"{dataset}_{model}_max-length_{max_length}_lr_{lr}_bs_{bs}_max-epochs_{max_epochs}_seed_{seed}"


def decode_exp_name(exp_name):
    dataset, model = exp_name.split("_")[:2]
    max_length, lr, bs, max_epochs, seed = exp_name.split("_")[3::2]
    max_length, lr, bs, max_epochs, seed = int(max_length), float(lr), int(bs), int(max_epochs), int(seed)
    return dataset, model, lr, bs, max_epochs, seed


def make_command(accumulate,
                 gpu_capacity,
                 lr,
                 bs,
                 max_steps,
                 seed,
                 data_dir,
                 results_dir,
                 check_int,
                 log_int,
                 ):

    exp_name = f"{encode_exp_name(lr, bs, max_steps, seed)}"
    
    if accumulate:
        accumulation = int(numpy.ceil(bs / gpu_capacity))
        bs_fill = gpu_capacity
    else:
        accumulation = 1
        bs_fill = bs

    command = (
        f'{os.path.join(repo_dir, "src", "main.py")} '
        f"--experiment {exp_name} "
        f"--batch_size {bs_fill} "
        f"--accumulate_int {accumulation} "
        f"--learning_rate {lr} "
        f"--training_steps {max_epochs} "
        f"--data_dir {data_dir} "
        f"--save_dir {results_dir} "
        f"--save_steps {check_int} "
        f"--verbose_steps {log_int} "
    )

    return command
