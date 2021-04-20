import torch
import argparse
import pdb
import time
import copy
import os
import torch_geometric.transforms as T
from dataset.DBLP import DBLP
from dataset.CoraML import CoraML
from dataset.Coauthor import Coauthor
from dataset.Amazon import Amazon
from models.Encoder import Encoder
from sklearn import metrics
from torch.optim import Adam
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE
from models.MyLinear import MyLinear


def train():
    # get the parameters
    args = get_args()
    print(args.domain)

    # decide the device
    device = torch.device('cuda:1' if torch.cuda.is_available() and args.cuda else 'cpu')

    # load dataset
    if args.domain == 'Cora':
        dataset = Planetoid(root='/home/amax/xsx/data/gnn_datas/Cora', name='Cora', transform=T.NormalizeFeatures())
    elif args.domain == 'CiteSeer':
        dataset = Planetoid(root='/home/amax/xsx/data/gnn_datas/CiteSeer', name='CiteSeer', transform=T.NormalizeFeatures())
    elif args.domain == 'PubMed':
        dataset = Planetoid(root='/home/amax/xsx/data/gnn_datas/PubMed', name='PubMed', transform=T.NormalizeFeatures())
    elif args.domain == 'DBLP':
        dataset = DBLP(root='/home/amax/xsx/data/gnn_datas/DBLP', name='DBLP')
    elif args.domain == 'Cora-ML':
        dataset = CoraML(root='/home/amax/xsx/data/gnn_datas/Cora_ML', name='Cora_ML')
    elif args.domain == 'CS':
        dataset = Coauthor(root='/home/amax/xsx/data/gnn_datas/Coauthor/CS', name='CS')
    elif args.domain == 'Physics':
        dataset = Coauthor(root='/home/amax/xsx/data/gnn_datas/Coauthor/Physics', name='Physics')
    elif args.domain == 'Computers':
        dataset = Amazon(root='/home/amax/xsx/data/gnn_datas/Amazon/Computers', name='Computers')
    elif args.domain == 'Photo':
        dataset = Amazon(root='/home/amax/xsx/data/gnn_datas/Amazon/Photo', name='Photo')
    else:
        dataset = None
        pdb.set_trace()
    data = dataset[0].to(device)

    # create the model and optimizer
    if args.model_type == "GAE":
        model = GAE(Encoder(data.num_features, args.hidden_dim, args.model_type)).to(device)
    elif args.model_type == "VGAE":
        model = VGAE(Encoder(data.num_features, args.hidden_dim, args.model_type)).to(device)
    else:
        print("the type of model is error!")
        model = None
        pdb.set_trace()
    optimizer = Adam(model.parameters(), lr=args.lr)

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
        loss = model.recon_loss(z, data.edge_index)
        if args.model_type == "VGAE":
            loss = loss + 0.001 * model.kl_loss()
        current_loss = loss.item()
        loss.backward()
        optimizer.step()

        # validate(model ,data)

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
    print("Total epochs: {:2d}".format(best_epoch + 100))
    print("Best epochs: {:2d}".format(best_epoch))
    # train a classification model
    node_classification(best_model, data, args, device, int(dataset.num_classes))
    print("Total time elapsed: {:.2f}s".format(used_time))


def node_classification(gae_model, data, args, device, num_classes):
    gae_model.eval()
    embeddings = gae_model.encode(data).detach()
    # with torch.no_grad():
    #     embeddings = gae_model.encode(data)

    model = MyLinear(args.hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # the information which need to be recorded
    least_loss = float("inf")
    best_model = None

    # beging training
    for epoch in range(args.nc_epochs):
        # the steps of training
        model.train()
        optimizer.zero_grad()
        output = model(embeddings)

        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        current_loss = loss.item()
        loss.backward()
        optimizer.step()

        # save the model if it access the minimum loss in current epoch
        if current_loss < least_loss:
            least_loss = current_loss
            best_model = copy.deepcopy(model)

    test(best_model, data, embeddings)


def validate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data)


def test(model, data, embeddings):
    model.eval()
    output = model(embeddings)
    pred = output[data.test_mask].max(1)[1].cpu().numpy()
    gold = data.y[data.test_mask].cpu().numpy()

    print("Test set results:",
          "accu= {:.2f}".format(metrics.accuracy_score(gold, pred) * 100),
          "macro_p= {:.2f}".format(metrics.precision_score(gold, pred, average='macro') * 100),
          "macro_r= {:.2f}".format(metrics.recall_score(gold, pred, average='macro') * 100),
          "macro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='macro') * 100),
          "micro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='micro') * 100))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Computers', help='which dataset to be used')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--nc_epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--model_type', type=str, default="GAE", help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(5):
        print("################  ", i, "  ################")
        train()
