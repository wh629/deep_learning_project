import json
import pandas as pd
import os
import argparse

from shared_settings import decode_exp_name


def collect_results(args):
    results = {
        "learning_rate": [],
        "batch_size": [],
        "max_steps": [],
        "seed": [],
        "best_val_loss" : [],
        "best_val_road" : [],
        "best_val_image" : [],
        "best_val_step" : [],
        "best_weights" : [],
        "current_step" : [],
        "max_step" : [],
    }

    def record_exp(one_exp_result):
        lr, bs, max_epochs, seed = decode_exp_name(one_exp_result["exp_name"])
        results["learning_rate"].append(lr)
        results["batch_size"].append(bs)
        results["max_steps"].append(max_epochs)
        results["seed"].append(seed)
        results["best_val_loss"].append(one_exp_result["best_val_loss"])
        results["best_val_road"].append(one_exp_result["best_val_road"])
        results["best_val_image"].append(one_exp_result["best_val_image"])
        results["best_val_step"].append(one_exp_result["best_val_step"])
        results["best_weights"].append(one_exp_result["best_weights"])
        results["current_step"].append(one_exp_result["current_step"])
        results["max_step"].append(one_exp_result["max_step"])
        results["early_stop"].append(one_exp_result["current_step"]<one_exp_result["max_step"])
	

    with open(os.path.join(args.results_dir, "results.jsonl"), "r") as reader:
        for row in reader:
            one_exp_result = json.loads(row)
            record_exp(one_exp_result)

    df_raw = pd.DataFrame.from_dict(results)
    df_raw.sort_values(by=["best_val_road", "best_val_image"], ascending=False, inplace=True)
    df_raw.to_csv(os.path.join(args.results_dir, "raw_results.csv"), index=False)
    
    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Collect results and make csv")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("DL_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("DL_DATA_DIR", os.path.join(repo_dir, "data"))
    )

    args = parser.parse_args()
    args.repo_dir = repo_dir
    collect_results(args)
