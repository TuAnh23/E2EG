"""
Code derived from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
"""
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
import argparse
from ogb.nodeproppred import PygNodePropPredDataset
from torch.optim import AdamW
from tqdm import tqdm


class BertClassifier(nn.Module):

    def __init__(self, num_classes, dropout, pretrain):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels):

        self.labels = labels
        self.texts = texts

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def train(model, train_X, train_y, val_X, val_y, learning_rate, epochs, batch_size, adam_epsilon):
    train, val = Dataset(train_X, train_y), Dataset(val_X, val_y)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_X): .3f} \
                | Train Accuracy: {total_acc_train / len(train_X): .3f} \
                | Val Loss: {total_loss_val / len(val_X): .3f} \
                | Val Accuracy: {total_acc_val / len(val_X): .3f}')


def evaluate(model, test_X, test_y):
    test = Dataset(test_X, test_y)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_y): .3f}')


def main():
    parser = argparse.ArgumentParser(description='Train BERT classifier baseline. Default hyper-params are similar '
                                     + 'to the BERT model used for GIANT.')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--data_root_dir', type=str, default='data')
    parser.add_argument('--raw-text-path', type=str, required=True,
                        help="Path of raw text (.txt file, each raw correspond to a node)")
    parser.add_argument('--truncate_length', type=int, default=128)
    parser.add_argument('--text_tokenizer_path', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=6e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrain', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    print(args)

    # Get raw text input
    with open(args.raw_text_path, "r") as fin:
        node_text_list = fin.readlines()

    # Tokenize the text
    tokenizer = BertTokenizer.from_pretrained(args.text_tokenizer_path)
    X = [tokenizer(text, padding='max_length', max_length=args.truncate_length,
                   truncation=True, return_tensors="pt")
         for text in node_text_list]
    torch.save(X, 'tokenized_text.pt')

    # Get targets from the standard OGB data
    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_root_dir)
    data = dataset[0]
    y = torch.flatten(data.y)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    model = BertClassifier(num_classes=40, dropout=args.dropout, pretrain=args.pretrain)

    train(model, [X[i] for i in train_idx], y[train_idx], [X[i] for i in valid_idx], y[valid_idx],
          args.learning_rate, args.epochs, args.batch_size, args.adam_epsilon)

    evaluate(model, [X[i] for i in test_idx], y[test_idx])


if __name__ == "__main__":
    main()
