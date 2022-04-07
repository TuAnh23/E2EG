import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Combine results from runs with different seeds')
    parser.add_argument('--experiment_dir', type=str, metavar="PATH", required=True)
    parser.add_argument('--runs', type=int, required=True)
    args = parser.parse_args()
    print(args)

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


if __name__ == "__main__":
    main()
