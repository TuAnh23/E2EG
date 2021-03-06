import argparse

import torch
import torch.nn.functional as F

import numpy as np

from logger import Logger
import numpy as np
from pecos.utils import smat_util

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_root_dir', type=str, default='../../dataset')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--node_emb_path', type=str, default=None)
    parser.add_argument('--save_pred', type=str, default="")
    args = parser.parse_args()
    print(args)

    dataset_name = args.data_dir.split('/')[-1]
    if dataset_name.endswith("_subset"):
        subset = True
        dataset_name = dataset_name.split("_")[0]
    else:
        subset = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if subset:
        x = torch.from_numpy(smat_util.load_matrix(args.node_emb_path).astype(np.float32))
        x = x.to(device)
        print("Loaded pre-trained node embeddings of shape={} from {}".format(x.shape, args.node_emb_path))
        y_true = np.load(args.data_dir + "/Y_main.all.npy")
        y_true = torch.tensor(np.expand_dims(y_true, axis=1)).to(device)
        split_idx = {}
        split_idx['train'] = torch.load(args.data_dir + "/train_idx.pt")
        split_idx['valid'] = torch.load(args.data_dir + "/valid_idx.pt")
        split_idx['test'] = torch.load(args.data_dir + "/test_idx.pt")
        train_idx = split_idx['train'].to(device)

        y_trn = np.load(args.data_dir + "/Y_main.trn.npy")
        nr_classes = np.max(y_trn) + 1
    else:
        dataset = PygNodePropPredDataset(name=dataset_name,root=args.data_root_dir)
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        if args.node_emb_path:
            data.x = torch.from_numpy(smat_util.load_matrix(args.node_emb_path).astype(np.float32))
            print("Loaded pre-trained node embeddings of shape={} from {}".format(data.x.shape, args.node_emb_path))

        x = data.x
        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            x = torch.cat([x, embedding], dim=-1)
        x = x.to(device)

        y_true = data.y.to(device)
        train_idx = split_idx['train'].to(device)

        nr_classes = dataset.num_classes


    model = MLP(x.size(-1), args.hidden_channels, nr_classes,
                args.num_layers, args.dropout).to(device)
    print(
        "Number of parameters: {}".format(
            sum(p.numel() for p in model.parameters())
        )
    )

    evaluator = Evaluator(name=dataset_name)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val = -1
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer)
            test_output = test(model, x, y_true, split_idx, evaluator)

            result = test_output[:-1]  # train, val, test accuracies
            logger.add_result(run, result)

            pred = test_output[-1]
            if result[1] > best_val:
                best_val = result[1]
                if args.save_pred:
                    print(f"Saving predictions at epoch {epoch}, run {run}.")
                    np.save(f"{args.save_pred}/y_pred_run{run}.npy", pred.cpu().detach().numpy())

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

    #     logger.print_statistics(run)
    # logger.print_statistics()
    print_statistics(logger, run)


def print_statistics(logger, run):
    result = 100 * torch.tensor(logger.results[run])
    argmax = result[:, 1].argmax().item()
    print(f'Val accuracy: {result[:, 1].max():.2f}')
    print(f'Train accuracy: {result[argmax, 0]:.2f}')
    print(f'Test accuracy: {result[argmax, 2]:.2f}')


if __name__ == "__main__":
    main()
