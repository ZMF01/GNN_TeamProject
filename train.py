import os
import argparse
import time
import numpy as np
import dgl
from dgl import DGLGraph
from sklearn.metrics import f1_score

from data_split import split_dataset
from models import GraphSage
from load import load_data
import torch as th
import torch.nn.functional as F
from utils import EarlyStopping, evaluate
from metrics import my_f1_score, accuracy



def main(args):
    # load data
    check_point_path = "./es_checkpoint_sage.pt"
    split_path = "./split.pt"
    if os.path.exists(check_point_path):
        os.remove(check_point_path)

    g, feats, labels, dim_features, n_classes = \
        load_data(args.dataset, True)
    G = dgl.from_networkx(g)
    print("Load data finished")

    # process command parameters
    print(f"{th.cuda.device_count()} GPUs")
    device = "cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu"
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # model, loss, optimizer
    model = GraphSage(in_dim=dim_features,
                 num_hidden=args.num_hidden_units,
                 n_classes=n_classes,
                 n_layers=args.n_layers,
                 activation=F.elu,
                 dropout=args.feat_drop,
                 aggregator_type=args.aggregator_type)
    print(model)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loss_fcn = F.nll_loss
    # loss_fcn = th.nn.BCEWithLogitsLoss()
    loss_fcn = th.nn.CrossEntropyLoss()

    # to device
    print("Begin Training")
    early_stopper = EarlyStopping(check_point_path, '<', args.patience)

    if args.fix_split and os.path.exists(split_path):
        train_mask, val_mask, test_mask = th.load(split_path)
    else:
        train_mask, val_mask, test_mask, _, _, _ = split_dataset(
        args.dataset,
        args.train_percent,
        args.val_percent)
        th.save([train_mask, val_mask, test_mask], split_path)

    if args.gpu >= 0:
        G = G.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        model = model.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    for epoch in range(args.num_epochs):
        model.train()

        train_logits = model(G, feats)
        train_logp = F.log_softmax(train_logits, 1)
        train_pred = train_logp.argmax(dim=1)
        train_loss = loss_fcn(train_logp[train_mask], labels[train_mask])
        train_f1_score = my_f1_score(labels, train_pred, train_mask, average='micro')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        val_score, val_loss = evaluate(G, feats, labels, val_mask, model, loss_fcn)
        test_score, test_loss = evaluate(G, feats, labels, test_mask, model, loss_fcn)

        print("Epoch {:05d} | Train Loss {:.4f} | Train F1-score {:.4f} | Val Loss {:.4f} | Val F1-score {:.6f} | "
              "Test Loss {:.4f} | Test F1-score {:.6f}".format(
            epoch, train_loss.item(), train_f1_score, val_loss, val_score, test_loss, test_score))

        if early_stopper.step(val_score, model):
            break

    model.load_state_dict(th.load(check_point_path))
    model.eval()
    test_score, test_loss = evaluate(G, feats, labels, test_mask, model, loss_fcn)

    print("Test Loss {:.4f} | Test F1-score {:.8f} on {}".format(test_loss, test_score, args.dataset))


# python train.py --gpu 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSage')
    # hardware related
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dataset", type=str, default="as_small",
                        help="The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit")
    # model related
    parser.add_argument("--concat", action="store_true", default=True,
                        help="concat neighbors with self")
    parser.add_argument("--num-hidden-units", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu, used in softmax of attention")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    # training related
    parser.add_argument("--patience", type=int, default=100, help="wait for how many iterations")
    parser.add_argument("--num-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="learning rate")
    parser.add_argument("--train_percent", type=float, default=0.05)
    parser.add_argument("--val_percent", type=float, default=0.2)
    parser.add_argument("--fix-split", action='store_true')
    # parser.add_argument('--batch-size', type=int, default='512', help='the size of each batch')

    args = parser.parse_args()
    print(args)

    main(args)
