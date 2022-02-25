
import argparse
import os
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.utils.featurization.text.preprocess import Preprocessor

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix


def main():
    parser = argparse.ArgumentParser(description='Prepare data for multi-task GIANT.')
    parser.add_argument('--raw-text-path', type=str, required=True, help="Path of raw text (.txt file, each raw correspond to a node)")
    parser.add_argument('--vectorizer-config-path', type=str, required=True, help="a path to a json file that specify the tfidf hyper-paramters")
    parser.add_argument('--data-root-dir', type=str, default="./dataset")
    parser.add_argument('--multi-task-data-dir', type=str, default="./proc_data_xrt")
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")
    parser.add_argument('--max-deg', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    # Change args.save_data_dir to args.save_data_dir/args.dataset
    save_data_dir = os.path.join(args.multi_task_data_dir, args.dataset)
    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_root_dir)
    data = dataset[0]
    edge_index = data.edge_index

    # Make sure edge_index is undirected!!!
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    # Filtering nodes whose number of edges >= max_degree
    Degree = degree(edge_index[0])
    Filtered_idx = torch.where(Degree < args.max_deg)[0]  # Add [:x] to the end to create a tiny dataset with x nodes
    print('Number of original nodes:{}'.format(data.x.shape[0]))
    print('Number of filtered nodes:{}'.format(len(Filtered_idx)))

    # Construct and save neighborhood-label matrix (adjacencey matrix) Y_neighbor.
    Y_csr_all = smat.csr_matrix(to_scipy_sparse_matrix(edge_index))
    Y_csr_trn = Y_csr_all[Filtered_idx]
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.trn.npz", Y_csr_trn)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.all.npz", Y_csr_all)
    print("Saved Y_neighbor.trn.npz and Y_neighbor.all.npz")

    # Save node main-label matrix
    Y_main_all = data.y.flatten().detach().numpy()
    Y_main_trn = Y_main_all[Filtered_idx]
    np.save(f"{save_data_dir}/Y_main.all", Y_main_all, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.trn", Y_main_trn, allow_pickle=False)
    print("Saved Y_main.all.npy and Y_main.trn.npy")

    # Apply the same filtering for raw text
    with open(args.raw_text_path, "r") as fin:
        node_text_list = fin.readlines()
    print("|node_text_list={}".format(len(node_text_list)))
    count = 0
    with open(f"{save_data_dir}/X.trn.txt", "w") as fout:
        for cur_idx, line in enumerate(node_text_list):
            if len(Filtered_idx) <= count:
                break
            if Filtered_idx[count].item() == cur_idx:
                fout.writelines(line)
                count += 1
    assert count == len(Filtered_idx), "count={}, len(Filtered_idx)={}".format(count, len(Filtered_idx))
    print("Saved X.trn.txt")

    # Apply the same filtering for tfidf features
    vectorizer_config = Vectorizer.load_config_from_args(args) # using args.vectorizer_config_path
    preprocessor = Preprocessor.train(node_text_list, vectorizer_config, dtype=np.float32)
    preprocessor.save(f"{save_data_dir}/tfidf-model")
    X_tfidf_all = preprocessor.predict(node_text_list)
    X_tfidf_trn = X_tfidf_all[Filtered_idx]
    smat_util.save_matrix(f"{save_data_dir}/X.all.tfidf.npz", X_tfidf_all)
    smat_util.save_matrix(f"{save_data_dir}/X.trn.tfidf.npz", X_tfidf_trn)
    print("Saved X.trn.npz and X.all.npz")


if __name__ == "__main__":
    main()
