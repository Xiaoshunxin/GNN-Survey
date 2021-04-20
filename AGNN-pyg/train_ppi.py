import torch
import argparse
import pdb
import os
import time
import copy
import torch
import torch_geometric.transforms as T
from models.AGNN import AGNN_PPI
from sklearn import metrics
from torch.optim import Adam
from torch.nn import functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader


def train():
    # get the parameters
    args = get_args()

    # decide the device
    device = torch.device('cuda:1' if torch.cuda.is_available() and args.cuda else 'cpu')

    # load dataset
    train_dataset = PPI(root='/home/amax/xsx/data/gnn_datas/PPI', split='train')
    val_dataset = PPI(root='/home/amax/xsx/data/gnn_datas/PPI', split='val')
    test_dataset = PPI(root='/home/amax/xsx/data/gnn_datas/PPI', split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    # data = dataset[0].to(device)

    # create the model and optimizer
    model = AGNN_PPI(train_dataset.num_features, args.hidden_dim, train_dataset.num_classes, args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # the information which need to be recorded
    start_time = time.time()
    bad_counter = 0
    best_valid_f1 = 0.0
    best_epoch = 0
    least_loss = float("inf")
    best_model = None

    # beging training
    for epoch in range(args.epochs):
        # the steps of training
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data.batch = None
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader.dataset)
        f1 = validate(model, val_loader, device)
        # print('Epoch: {:04d}'.format(epoch + 1), 'f1: {:.4f}'.format(f1), 'loss: {:.4f}'.format(avg_loss))

        if avg_loss < least_loss:
            least_loss = avg_loss
            best_epoch = epoch + 1
            best_valid_f1 = f1
            best_model = copy.deepcopy(model)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            break

    print("Optimization Finished!")
    used_time = time.time() - start_time
    print("Total epochs: {:2d}".format(best_epoch + 100))
    print("Best epochs: {:2d}".format(best_epoch))
    # print("Time for each epoch: {:.2f}s".format(used_time / (best_epoch + args.patience)))
    print("Best epoch's validate f1: {:.2f}".format(best_valid_f1 * 100))
    # test the best trained model
    test(best_model, test_loader, device)
    print("Total time elapsed: {:.2f}s".format(used_time))

def validate(model, data_loader, device):
    model.eval()

    ys, preds = [], []
    for data in data_loader:
        ys.append(data.y)  # data.y shape: (num_nodes, num_classes)
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        preds.append((output > 0).float().cpu())
    gold, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()  # shape: (num_nodes, num_class)
    return metrics.f1_score(gold, pred, average='macro')


def test(model, data_loader, device):
    model.eval()

    ys, preds = [], []
    for data in data_loader:
        ys.append(data.y)
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        preds.append((output > 0).float().cpu())
    gold, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    print("Test set results:",
          "accu= {:.2f}".format(metrics.accuracy_score(gold, pred) * 100),
          "macro_p= {:.2f}".format(metrics.precision_score(gold, pred, average='macro') * 100),
          "macro_r= {:.2f}".format(metrics.recall_score(gold, pred, average='macro') * 100),
          "macro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='macro') * 100),
          "micro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='micro') * 100))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Cora', help='which dataset to be used')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(5):
        print("################  ", i, "  ################")
        train()