import argparse
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Combine results from runs with different seeds')
    parser.add_argument('--experiment_dir', type=str, metavar="PATH", required=True)
    parser.add_argument('--runs', type=int, required=True)
    args = parser.parse_args()
    print(args)

    best_valid_accs = []
    train_accs = []
    test_accs = []

    for run in range(0, args.runs):
        print(f"------------------ Results run {run} ------------------")
        experiment_dir = args.experiment_dir + f"/run{run}"

        # Extract validation scores
        with open(f"{experiment_dir}/val_performance_per_round.log", "r") as file:
            val_strs = file.readlines()
        val_strs = [x for x in val_strs if x.startswith("Final performance")]
        val_score_per_round = [float(re.search('val_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
        best_val_acc = max(val_score_per_round)
        best_val_index = max(range(len(val_score_per_round)), key=lambda i: val_score_per_round[i])
        best_valid_accs.append(best_val_acc)
        print(f"Best validation accuracy {best_val_acc}" + \
              f" at round {best_val_index}")

        # Extract train scores
        train_score_per_round = [float(re.search('train_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
        train_accs.append(train_score_per_round[best_val_index])
        print(f"Train accuracy: {train_score_per_round[best_val_index]}")

        # Extract test scores
        with open(f"{experiment_dir}/test_scores.txt", "r") as file:
            test_strs = file.readlines()
        test_strs = [x for x in test_strs if x.startswith("\t")]
        test_score_per_round = [float(re.search('Multi-class accuracy: (.+?)\n', text).group(1)) for text in test_strs]
        test_accs.append(test_score_per_round[best_val_index])
        print(f"Test accuracy: {test_score_per_round[best_val_index]}")

    print("------------------ Overall result ------------------")
    print(f'Highest Valid: {np.array(best_valid_accs).mean()} ± {np.array(best_valid_accs).std()}')
    print(f'Final Train: {np.array(train_accs).mean()} ± {np.array(train_accs).std()}')
    print(f'Final Test: {np.array(test_accs).mean()} ± {np.array(test_accs).std()}')


if __name__ == "__main__":
    main()
