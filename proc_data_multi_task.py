
import argparse
import gc
import os
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.utils.featurization.text.preprocess import Preprocessor

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, is_undirected, to_undirected, k_hop_subgraph
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Prepare data for multi-task GIANT.')
    parser.add_argument('--raw-text-path', type=str, required=True, help="Path of raw text (.txt file, each raw correspond to a node)")
    parser.add_argument('--vectorizer-config-path', type=str, required=True, help="a path to a json file that specify the tfidf hyper-paramters")
    parser.add_argument('--data-root-dir', type=str, default="./dataset")
    parser.add_argument('--multi-task-data-dir', type=str, default="./proc_data_xrt")
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")
    parser.add_argument('--subset', type=str, default="", help="Pass in '_subset' to take a subset, otherwise empty string''")
    parser.add_argument('--max-deg', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    # Change args.save_data_dir to args.save_data_dir/args.dataset
    save_data_dir = os.path.join(args.multi_task_data_dir, args.dataset + args.subset)

    if not os.path.isdir(save_data_dir):
        # Create directory to store data if not yet exist
        os.mkdir(save_data_dir)

    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_root_dir)
    data = dataset[0]
    edge_index = data.edge_index

    if args.subset:
        node_subset, edge_index, _, _ = k_hop_subgraph(node_idx=torch.tensor(range(0, 2000, 100)),
                                                       num_hops=2,
                                                       edge_index=data.edge_index)
        # Re-index the nodes to be in [0, nr_nodes-1]
        map_old_new_idx = dict(zip(node_subset.tolist(), range(0, len(node_subset))))
        edge_index = edge_index.apply_(lambda x: map_old_new_idx[x])

    # Make sure edge_index is undirected!!!
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    # Filtering nodes whose number of edges >= max_degree
    Degree = degree(edge_index[0])
    Filtered_idx = torch.where(Degree < args.max_deg)[0]
    print('Number of original nodes:{}'.format(data.x.shape[0]))
    if args.subset:
        print('Number of subset nodes:{}'.format(len(node_subset)))
    print('Number of filtered nodes:{}'.format(len(Filtered_idx)))

    # Split the data to train-val-test
    if args.subset:
        # Preserve the split ratio of the original data
        split_idx = dataset.get_idx_split()
        train_ratio = len(split_idx["train"]) / data.x.shape[0]
        validation_ratio = len(split_idx["valid"]) / data.x.shape[0]
        test_ratio = len(split_idx["test"]) / data.x.shape[0]

        train_val_idx, test_idx = train_test_split(range(0, len(node_subset)), train_size=train_ratio + validation_ratio, random_state=0)
        train_idx, valid_idx = train_test_split(train_val_idx, train_size=train_ratio/(train_ratio+validation_ratio), random_state=0)

        train_idx = torch.tensor(train_idx)
        valid_idx = torch.tensor(valid_idx)
        test_idx = torch.tensor(test_idx)

        torch.save(train_idx, f"{save_data_dir}/train_idx.pt")
        torch.save(valid_idx, f"{save_data_dir}/valid_idx.pt")
        torch.save(test_idx, f"{save_data_dir}/test_idx.pt")

    else:
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    # Apply the same filtering
    train_idx = np.intersect1d(train_idx.cpu().numpy(), Filtered_idx.cpu().numpy())  # Add [:x] to the end to create a tiny dataset with x nodes
    valid_idx = np.intersect1d(valid_idx.cpu().numpy(), Filtered_idx.cpu().numpy())
    test_all_idx = test_idx.cpu().numpy()
    test_idx = np.intersect1d(test_all_idx, Filtered_idx.cpu().numpy())

    print('Number of train nodes:{}'.format(len(train_idx)))
    print('Number of val nodes:{}'.format(len(valid_idx)))
    print('Number of test nodes:{}'.format(len(test_all_idx)))


    # Construct and save neighborhood-label matrix (adjacencey matrix) Y_neighbor.
    Y_csr_all = smat.csr_matrix(to_scipy_sparse_matrix(edge_index))
    Y_csr_trn = Y_csr_all[train_idx]
    Y_csr_val = Y_csr_all[valid_idx]
    Y_csr_test = Y_csr_all[test_idx]
    Y_csr_test_all = Y_csr_all[test_all_idx]

    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.trn.npz", Y_csr_trn)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.val.npz", Y_csr_val)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.test.npz", Y_csr_test)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.test_all.npz", Y_csr_test_all)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.all.npz", Y_csr_all)
    print("Saved Y_neighbor.trn.npz, Y_neighbor.val.npz, Y_neighbor.test.npz, Y_neighbor.test_all.npz and Y_neighbor.all.npz")

    # Save node main-label matrix
    Y_main_all = data.y.flatten().detach().numpy()
    if args.subset:
        Y_main_all = Y_main_all[node_subset]
    Y_main_trn = Y_main_all[train_idx]
    Y_main_val = Y_main_all[valid_idx]
    Y_main_test = Y_main_all[test_idx]
    Y_main_test_all = Y_main_all[test_all_idx]
    np.save(f"{save_data_dir}/Y_main.all", Y_main_all, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.trn", Y_main_trn, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.val", Y_main_val, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.test", Y_main_test, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.test_all", Y_main_test_all, allow_pickle=False)
    print("Saved Y_main.trn.npy, Y_main.val.npy, Y_main.test.npy, Y_main.test_all and Y_main.all.npy")

    # Apply the same filtering for raw text
    with open(args.raw_text_path, "r") as fin:
        node_text_list = fin.readlines()
    print("|node_text_list={}".format(len(node_text_list)))

    if args.subset:
        filter_list(node_text_list, node_subset, save_path=f"{save_data_dir}/X.all.txt")
        del node_text_list
        gc.collect()

        with open(f"{save_data_dir}/X.all.txt", "r") as fin:
            node_text_list = fin.readlines()
        print("|subset node_text_list={}".format(len(node_text_list)))

    filter_list(node_text_list, train_idx, save_path=f"{save_data_dir}/X.trn.txt")
    print("Saved X.trn.txt")
    filter_list(node_text_list, valid_idx, save_path=f"{save_data_dir}/X.val.txt")
    print("Saved X.val.txt")
    filter_list(node_text_list, test_idx, save_path=f"{save_data_dir}/X.test.txt")
    print("Saved X.test.txt")
    filter_list(node_text_list, test_all_idx, save_path=f"{save_data_dir}/X.test_all.txt")
    print("Saved X.test_all.txt")

    # Apply the same filtering for tfidf features
    vectorizer_config = Vectorizer.load_config_from_args(args) # using args.vectorizer_config_path
    preprocessor = Preprocessor.train(node_text_list, vectorizer_config, dtype=np.float32)
    preprocessor.save(f"{save_data_dir}/tfidf-model")
    X_tfidf_all = preprocessor.predict(node_text_list)
    X_tfidf_trn = X_tfidf_all[train_idx]
    X_tfidf_val = X_tfidf_all[valid_idx]
    X_tfidf_test = X_tfidf_all[test_idx]
    X_tfidf_test_all = X_tfidf_all[test_all_idx]
    smat_util.save_matrix(f"{save_data_dir}/X.all.tfidf.npz", X_tfidf_all)
    smat_util.save_matrix(f"{save_data_dir}/X.trn.tfidf.npz", X_tfidf_trn)
    smat_util.save_matrix(f"{save_data_dir}/X.val.tfidf.npz", X_tfidf_val)
    smat_util.save_matrix(f"{save_data_dir}/X.test.tfidf.npz", X_tfidf_test)
    smat_util.save_matrix(f"{save_data_dir}/X.test_all.tfidf.npz", X_tfidf_test_all)
    print("Saved X.trn.tfidf.npz, X.val.tfidf.npz, X.test.tfidf.npz, X.test_all.tfidf.npz and X.all.tfidf.npz")


def filter_list(original_list, indices, save_path):
    """
    @param original_list: the full list
    @param indices: the index of items that we want to keep
    @param save_path: the path to save the new filtered list
    """
    with open(save_path, "w") as fout:
        for i in indices:
            fout.writelines(original_list[i])


if __name__ == "__main__":
    main()
