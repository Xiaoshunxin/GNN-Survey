import torch
import argparse
import pdb
import time
import copy
import torch_geometric.transforms as T
from models.Encoder_LP import Encoder
from sklearn import metrics
from torch.optim import Adam
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE


def train():
    # get the parameters
    args = get_args()

    # decide the device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # load dataset
    if args.domain == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    elif args.domain == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', transform=T.NormalizeFeatures())
    elif args.domain == 'PubMed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=T.NormalizeFeatures())
    else:
        dataset = None
    if dataset is None:
        pdb.set_trace()
    data = dataset[0]

    # create the model and optimizer
    if args.model_type == "GAE":
        model = GAE(Encoder(data.num_features, args.output_dim, args.model_type)).to(device)
    elif args.model_type == "VGAE":
        model = VGAE(Encoder(data.num_features, args.output_dim, args.model_type)).to(device)

    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = model.split_edges(data).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # the information which need to be recorded
    start_time = time.time()
    bad_counter = 0
    best_epoch = 0
    least_loss = float("inf")
    best_model = None

    # beging training
    for epoch in range(args.epochs):
        # the steps of training
        model.train()
        optimizer.zero_grad()

        z = model.encode(data)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        if args.model_type == "VGAE":
            loss = loss + 0.001 * model.kl_loss()
        current_loss = loss.item()
        loss.backward()
        optimizer.step()

        # validate current model
        ap = validate(model, data)
        print('Epoch: {:04d}'.format(epoch + 1), 'ap: {:.4f}'.format(ap))

        # save the model if it access the minimum loss in current epoch
        if current_loss < least_loss:
            least_loss = current_loss
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)
            bad_counter = 0
        else:
            bad_counter += 1

        # early stop
        if bad_counter >= args.patience:
            break

    print("Optimization Finished!")
    used_time = time.time() - start_time
    print("Total time elapsed: {:.2f}s".format(used_time))
    print("Time for each epoch: {:.2f}s".format(used_time / (best_epoch + args.patience)))
    print("Best validate f1 in: {:2d}".format(best_epoch), " with value: {:.2f}".format(best_valid_ap * 100))

    # test the best trained model
    test(best_model, data)

def validate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data)
    roc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    return ap



def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data)
    roc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    print("Test set results:",
          "roc= {:.2f}".format(roc * 100),
          "ap= {:.2f}".format(ap * 100))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Cora', help='which dataset to be used')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--output_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--model_type', type=str, default="VGAE", help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    return parser.parse_args()


if __name__ == '__main__':
    train()