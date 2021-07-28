import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import random
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import os

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from logger import Logger
from model import GMLP,GATE,GMLP_SLE
from utils import set_seed

def gmlp_train(model, loader, optimizer, device, epoch, epochs):
    model.train()

    total_loss = 0
    loss_fcn =nn.CrossEntropyLoss()
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(torch.long).to(device)
        optimizer.zero_grad()
        output_att,output_concat = model(xs)
        L1 = loss_fcn(output_att,  y.squeeze(1))
        L2 = loss_fcn(output_concat,  y.squeeze(1))
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
    return total_loss / len(loader.dataset)

def train(model, loader, optimizer, device, epoch, epochs):
    model.train()

    total_loss = 0
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(torch.long).to(device)
        optimizer.zero_grad()

        output_att = model(xs)
        L1 = F.cross_entropy(output_att, y.squeeze(1))
        loss = L1

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
    return total_loss / len(loader.dataset)

@torch.no_grad()
def gmlp_test(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y_true.append(y.to(torch.long))
        output = model(xs)[0]
        y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())

    return evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    

@torch.no_grad()
def test(model, loader, evaluator, device):
    model.eval()

    y_true, y_pred = [], []
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y_true.append(y.to(torch.long))
        output = model(xs)
        y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())

    return evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=6)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--n_layers_1', type=int, default=4)
    parser.add_argument('--n_layers_2', type=int, default=6)
    parser.add_argument('--n_layers_3', type=int, default=4)    
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pre_process', action='store_true')
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = PygNodePropPredDataset('ogbn-papers100M', root='/home2/zwt/')

    split_idx = dataset.get_idx_split()
    data = dataset[0]
    num_feat = data.x.shape[1]
    num_classes = int((data.y.data.to(torch.long).max()+1).item())

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.xs = []
            idx = split_idx[split]

            N = data.num_nodes

            t = time.perf_counter()
            print(f'Reading {split} node features...', end=' ', flush=True)

            x = data.x[idx]
            #self.xs.append(x.float())
#            self.label_emb = torch.load(f'./dgl_y_{split}.pt')
            for i in range(args.num_hops + 1):
                x = torch.load(f'/home2/zwt/6-yinziqi/gmlp_test/x_{split}_{i}.pt')
                self.xs.append(x.float())
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            self.y = (data.y.data)[idx].to(torch.long)

        def __len__(self):
            return self.xs[0].size(0)

        def __getitem__(self, idx):
            return [x[idx] for x in self.xs], self.y[idx]

    train_dataset = MyDataset(split='train')
    valid_dataset = MyDataset(split='valid')
    test_dataset = MyDataset(split='test')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=6, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)	
    '''
    model = GMLP(num_feat,label_feat, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,4,args.pre_process).to(device)              
    '''
    evaluator = Evaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        set_seed(run)
        '''
        model = GMLP2(num_feat,label_feat, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.pre_process).to(device)
        '''
        model = GMLP(num_feat, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.pre_process).to(device)
        print(model)
        '''                 
        model = GMLP_SLE(num_feat,num_classes, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.n_layers_3,args.pre_process).to(device)
        '''
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.
        valid_test_acc = 0.
        for epoch in range(args.epochs+1):
            t = time.perf_counter()
            train_loss = gmlp_train(model, train_loader, optimizer, device, epoch, args.epochs)
            if epoch >= 99:
                train_acc = gmlp_test(model, train_loader, evaluator, device)
                valid_acc = gmlp_test(model, valid_loader, evaluator, device)
                test_acc = gmlp_test(model, test_loader, evaluator, device)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    if not os.path.isdir('./checkpoint'):
                        os.mkdir('./checkpoint')
                    torch.save(model, './checkpoint/gmlp.pkl')

                logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                if epoch >= 99:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch + 1:02d}, '
                          f'Training loss: {train_loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%, '
                          f'Best: {100 * valid_test_acc:.2f}%, '
                          f'Time: {time.perf_counter()-t:.4f}')
                else:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch + 1:02d}, '
                          f'Training loss: {train_loss:.4f}, '
                          f'Time: {time.perf_counter()-t:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()

