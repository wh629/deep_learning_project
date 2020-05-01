import os
import numpy

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_exp_name(lr, bs, max_steps, seed):
    return f"lr_{lr}_bs_{bs}_max-steps_{max_steps}_seed_{seed}"


def decode_exp_name(exp_name):
    dataset, model = exp_name.split("_")[:2]
    lr, bs, max_steps, seed = exp_name.split("_")[3::2]
    lr, bs, max_steps, seed = float(lr), int(bs), int(max_steps), int(seed)
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
        f"--road_lambda {road_lambda} "
        f"--box_lambda {box_lambda} "
    )

    return command
