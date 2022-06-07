import argparse
import numpy as np
import re


def main():
    parser = argparse.ArgumentParser(description='Combine results from runs with different seeds')
    parser.add_argument('--experiment_dir', type=str, metavar="PATH", required=True)
    parser.add_argument('--runs', type=int, required=True)
    args = parser.parse_args()
    print(args)

    scores_per_round(args.experiment_dir, args.runs, f"{args.experiment_dir}/scores_per_round.txt")

    val_accs = []
    train_accs = []
    test_accs = []

    for run in range(0, args.runs):
        with open(args.experiment_dir + f"/run{run}/final_scores.txt", "r") as file:
            final_result_per_run = file.readlines()
        for line in final_result_per_run:
            if line.startswith("Train accuracy"):
                train_accs.append(float(line.split()[-1]))
            elif line.startswith("Val accuracy"):
                val_accs.append(float(line.split()[-1]))
            elif line.startswith("Test accuracy"):
                test_accs.append(float(line.split()[-1]))

    print("------------------ Overall result ------------------")
    print(f'Highest Valid: {np.array(val_accs).mean()} ± {np.array(val_accs).std()}')
    print(f'Final Train: {np.array(train_accs).mean()} ± {np.array(train_accs).std()}')
    print(f'Final Test: {np.array(test_accs).mean()} ± {np.array(test_accs).std()}')


def scores_per_round(experiment_dir, runs, out_file):
    val_scores = []
    train_scores = []
    for run in range(0, runs):
        run_dir = f"{experiment_dir}/run{run}"
        # Extract validation scores
        with open(f"{run_dir}/val_performance_per_round.log", "r") as file:
            val_strs = file.readlines()
        val_strs = [x for x in val_strs if x.startswith("Final performance")]
        val_score_per_round = [float(re.search('val_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
        val_scores.append(val_score_per_round)

        # Extract train scores
        train_score_per_round = [float(re.search('train_acc_mclass=(.+?),', text).group(1)) for text in val_strs]
        train_scores.append(train_score_per_round)

    val_scores = np.array(val_scores)
    train_scores = np.array(train_scores)
    output = ""
    for round_i in range(0, val_scores.shape[1]):
        output = output + f"Round {round_i}: \n" + \
            f"Train acc: {train_scores[:,round_i].mean()} ± {train_scores[:,round_i].std()}\n" + \
            f"Val acc: {val_scores[:, round_i].mean()} ± {val_scores[:, round_i].std()}\n\n"

    with open(out_file, 'w') as f:
        f.write(output)


if __name__ == "__main__":
    main()
