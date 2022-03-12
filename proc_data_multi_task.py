
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

    # Split the data to train-val-test
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    # Apply the same filtering
    train_idx = np.intersect1d(train_idx.cpu().numpy(), Filtered_idx.cpu().numpy())  # Add [:x] to the end to create a tiny dataset with x nodes
    valid_idx = np.intersect1d(valid_idx.cpu().numpy(), Filtered_idx.cpu().numpy())
    test_idx = np.intersect1d(test_idx.cpu().numpy(), Filtered_idx.cpu().numpy())

    print('Number of train nodes:{}'.format(len(train_idx)))
    print('Number of val nodes:{}'.format(len(valid_idx)))
    print('Number of test nodes:{}'.format(len(test_idx)))


    # Construct and save neighborhood-label matrix (adjacencey matrix) Y_neighbor.
    Y_csr_all = smat.csr_matrix(to_scipy_sparse_matrix(edge_index))
    Y_csr_trn = Y_csr_all[train_idx]
    Y_csr_val = Y_csr_all[valid_idx]
    Y_csr_test = Y_csr_all[test_idx]

    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.trn.npz", Y_csr_trn)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.val.npz", Y_csr_val)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.test.npz", Y_csr_test)
    smat_util.save_matrix(f"{save_data_dir}/Y_neighbor.all.npz", Y_csr_all)
    print("Saved Y_neighbor.trn.npz, Y_neighbor.val.npz, Y_neighbor.test.npz and Y_neighbor.all.npz")

    # Save node main-label matrix
    Y_main_all = data.y.flatten().detach().numpy()
    Y_main_trn = Y_main_all[train_idx]
    Y_main_val = Y_main_all[valid_idx]
    Y_main_test = Y_main_all[test_idx]
    np.save(f"{save_data_dir}/Y_main.all", Y_main_all, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.trn", Y_main_trn, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.val", Y_main_val, allow_pickle=False)
    np.save(f"{save_data_dir}/Y_main.test", Y_main_test, allow_pickle=False)
    print("Saved Y_main.trn.npy, Y_main.val.npy, Y_main.test.npy and Y_main.all.npy")

    # Apply the same filtering for raw text
    with open(args.raw_text_path, "r") as fin:
        node_text_list = fin.readlines()
    print("|node_text_list={}".format(len(node_text_list)))

    filter_list(node_text_list, train_idx, save_path=f"{save_data_dir}/X.trn.txt")
    print("Saved X.trn.txt")
    filter_list(node_text_list, valid_idx, save_path=f"{save_data_dir}/X.val.txt")
    print("Saved X.val.txt")
    filter_list(node_text_list, test_idx, save_path=f"{save_data_dir}/X.test.txt")
    print("Saved X.test.txt")

    # Apply the same filtering for tfidf features
    vectorizer_config = Vectorizer.load_config_from_args(args) # using args.vectorizer_config_path
    preprocessor = Preprocessor.train(node_text_list, vectorizer_config, dtype=np.float32)
    preprocessor.save(f"{save_data_dir}/tfidf-model")
    X_tfidf_all = preprocessor.predict(node_text_list)
    X_tfidf_trn = X_tfidf_all[train_idx]
    X_tfidf_val = X_tfidf_all[valid_idx]
    X_tfidf_test = X_tfidf_all[test_idx]
    smat_util.save_matrix(f"{save_data_dir}/X.all.tfidf.npz", X_tfidf_all)
    smat_util.save_matrix(f"{save_data_dir}/X.trn.tfidf.npz", X_tfidf_trn)
    smat_util.save_matrix(f"{save_data_dir}/X.val.tfidf.npz", X_tfidf_val)
    smat_util.save_matrix(f"{save_data_dir}/X.test.tfidf.npz", X_tfidf_test)
    print("Saved X.trn.tfidf.npz, X.val.tfidf.npz, X.test.tfidf.npz and X.all.tfidf.npz")


def filter_list(list, index, save_path):
    """
    @param list: the full list
    @param index: the index of items that we want to keep
    @param save_path: the path to save the new filtered list
    """
    count = 0
    with open(save_path, "w") as fout:
        for cur_idx, line in enumerate(list):
            if len(index) <= count:
                break
            if index[count].item() == cur_idx:
                fout.writelines(line)
                count += 1
    assert count == len(index), "count={}, len(Filtered_idx)={}".format(count, len(index))


if __name__ == "__main__":
    main()
