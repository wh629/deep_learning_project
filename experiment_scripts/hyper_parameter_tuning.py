"""
Hyper parameter tuning
"""

import random
import os
import argparse
from shared_settings import make_command


def select_candidates():
    lr_candidates = [1e-2, 5e-2, 1e-3]
    bs_candidates = [8, 16]
    max_steps_candidates = [1e3, 5e3, 1e4]
    seed_range = 1e6
    lr = lr_candidates[random.randrange(0, len(lr_candidates), 1)]
    bs = bs_candidates[random.randrange(0, len(bs_candidates), 1)]
    max_steps = max_epochs_candidates[random.randrange(0, len(max_epochs_candidates), 1)]
    seed = random.randint(0, seed_range)

    return lr, bs, max_epochs, seed


def submit_trials(args):
    jobs = []

    for trial in range(args.n_trials):
        # select candidates for trial
        lr, bs, max_steps, seed = select_candidates()
        command = make_command(
            args.accumulate,       # whether to do accumulating gradients
            args.gpu_capacity,     # maximum batch size per gpu
            lr,                    # learning rate
            bs,                    # batch size
            max_steps,             # maximum number of update steps
            seed,                  # seed
            args.data_dir,         # data directory
            args.results_dir,      # results directory
            args.check_int,        # checking interval
            args.log_int,          # logging interval
            args.road_lambda,      # relative weight of road loss
            args.box_lambda,       # relative weight of box loss
        )
        sbatch_file = os.path.join(args.repo_dir, "experiment_scripts", f"{args.user}.sbatch")
        jobs.append(f'COMMAND="{command}" sbatch {sbatch_file}\n')

    with open(os.path.join(args.repo_dir, "experiment_scripts", "submit_sbatch.sh"), "a") as f:
        for one_job in jobs:
            f.write(one_job)

    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Run Deep Learning Experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("DL_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("DL_DATA_DIR", os.path.join(repo_dir, "data"))
    )
    parser.add_argument("--user", type=str)

    parser.add_argument("--n-trials", type=int, help="number of trials")

    parser.add_argument("--gpu-capacity", type=int, default=2)
    parser.add_argument('--check_int', type=int, default=1000)
    parser.add_argument('--log_int',type=int, default=100)
    parser.add_argument("--accumulate", action='store_true')
    parser.add_argument('--road_lambda', type=float, default=1.0)
    parser.add_argument('--box_lambda', type=float, default=1.0)

    args = parser.parse_args()
    args.repo_dir = repo_dir
    submit_trials(args)
