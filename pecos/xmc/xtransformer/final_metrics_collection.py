import argparse
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Collect final metrics: best validation accuracy and the corresponding'
                                                 ' test accuracy')
    parser.add_argument('--experiment_dir', type=str, metavar="PATH", required=True)
    parser.add_argument(
        "--wandb-username",
        type=str,
        default=None,
        help="Username if want to log results to wandb. If not passed, do not use wandb",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Run id to continue logging results to wandb. Should be the same as the one used in the training script",
    )
    args = parser.parse_args()
    print(args)

    experiment_dir = args.experiment_dir

    best_val_acc, best_val_index, final_train_acc = extract_train_performance_logs(experiment_dir)

    final_test_acc = extract_test_performance(experiment_dir)

    if args.wandb_username is not None:
        import wandb
        wandb.init(project="UvA_Thesis", entity=args.wandb_username, id=args.wandb_run_id, resume="must")
        wandb.log({"best_round": best_val_index,
                   "final_train_acc": final_train_acc,
                   "final_val_acc": best_val_acc,
                   "final_test_acc": final_test_acc}
                  )

        wandb.run.summary["best_round"] = best_val_index
        wandb.run.summary["final_train_acc"] = final_train_acc
        wandb.run.summary["final_val_acc"] = best_val_acc
        wandb.run.summary["final_test_acc"] = final_test_acc


def extract_test_performance(experiment_dir):
    # Extract test score
    with open(f"{experiment_dir}/test_scores.txt", "r") as file:
        test_strs = file.readlines()
    test_strs = [x for x in test_strs if x.startswith("\t")]
    final_test_acc = [float(re.search('Multi-class accuracy: (.+?)\n', text).group(1)) for text in test_strs][0]
    print(f"Test accuracy: {final_test_acc}")
    return final_test_acc


def extract_train_performance_logs(experiment_dir):
    # Extract validation scores
    with open(f"{experiment_dir}/val_performance_per_round.log", "r") as file:
        val_strs = file.readlines()
    val_strs = [x for x in val_strs if x.startswith("Final performance")]
    val_score_per_round = [float(re.search('val_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
    best_val_acc = max(val_score_per_round)
    best_val_index = max(range(len(val_score_per_round)), key=lambda i: val_score_per_round[i])
    print(f"Best validation accuracy at round {best_val_index}")
    print(f"Val accuracy: {best_val_acc}")

    # Extract train scores
    train_score_per_round = [float(re.search('train_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
    final_train_acc = train_score_per_round[best_val_index]
    print(f"Train accuracy: {final_train_acc}")
    return best_val_acc, best_val_index, final_train_acc


if __name__ == "__main__":
    main()
