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

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))            
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout,alpha):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.in_feats=in_feats
        self.hidden=hidden
        self.out_feats=out_feats
        if n_layers == 1:
            self.layers.append(Dense(in_feats, out_feats))
        else:
            self.layers.append(Dense(in_feats, hidden))
            #middle
            for i in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden,alpha))
            #end
            self.layers.append(Dense(hidden, out_feats))                       
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, x):
        x=self.layers[0](x)
        h0=x
        for layer_id, layer in enumerate(self.layers):
            if layer_id==0:
                continue
            elif layer_id== self.n_layers - 1:
                x = layer(x)                 
            else:
                x = layer(x,h0)
                x = self.dropout(self.prelu(x))                
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,alpha):
        super(GraphConvolution, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.alpha=alpha
        self.reset_parameters()
        self.bias = nn.BatchNorm1d(out_features)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input ,h0):    
        support = (1-self.alpha)*input+self.alpha*h0      
        output = torch.mm(support, self.weight)
        output=self.bias(output)        
        if self.in_features==self.out_features:
            output = output+input
        return output

class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

class GMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2):
        super(GMLP, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        self.lr_left1 = FeedForwardNetII(hidden*num_hops, hidden, hidden, n_layers_1-1, dropout,alpha)
        self.lr_left2 = nn.Linear(hidden, nclass) 
        self.lr_att = nn.Linear(hidden + hidden, 1)
        self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
        self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.reset_parameters()
    def reset_parameters(self):
        self.lr_left1.reset_parameters()
        self.lr_left2.reset_parameters()
        self.lr_att.reset_parameters()
        self.lr_right1.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]   
              
        drop_features = [self.input_drop(feature) for feature in feature_list]
        for i in range(len(drop_features)):
            drop_features[i] = self.fcs[i](drop_features[i])
        concat_features = torch.cat(drop_features, dim=1)

        left_1 = self.dropout(self.prelu(self.lr_left1(concat_features)))
        left_2 = self.lr_left2(left_1)
        
        attention_scores = [torch.sigmoid(self.lr_att(torch.cat((left_1, x), dim=1))).view(num_node, 1) for x in
                            drop_features]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(drop_features[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(drop_features[i],self.att_drop(W[:, i].view(num_node, 1)))
        right_1 = self.lr_right1(right_1)        
        
        return right_1, left_2


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_first: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_first:
                x = batch_norm(x.relu_())
            else:
                x = batch_norm(x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class SIGN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_embeddings: int, num_layers: int,
                 dropout: float = 0.0, batch_norm: bool = True,
                 relu_first: bool = False):
        super(SIGN, self).__init__()

        self.mlps = ModuleList()
        for _ in range(num_embeddings):
            mlp = MLP(in_channels, hidden_channels, hidden_channels,
                      num_layers, dropout, batch_norm, relu_first)
            self.mlps.append(mlp)

        self.mlp = MLP(num_embeddings * hidden_channels, hidden_channels,
                       out_channels, num_layers, dropout, batch_norm,
                       relu_first)
        self.reset_parameters()  
    
    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, xs):
        out = []
        for x, mlp in zip(xs, self.mlps):
            out.append(mlp(x))
        out = torch.cat(out, dim=-1).relu_()
        return self.mlp(out)

def custom_cross_entropy(x, labels):
    epsilon = 1 - math.log(2)
    loss = F.cross_entropy(x, labels.squeeze(1))
    loss = torch.log(epsilon + loss) - math.log(epsilon)
    return torch.mean(loss)

def train(model, loader, optimizer, device, epoch, epochs):
    model.train()

    total_loss = 0
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(torch.long).to(device)
        optimizer.zero_grad()

        output_att, output_concat = model(xs)
        L1 = F.cross_entropy(output_att, y.squeeze(1))
        L2 = F.cross_entropy(output_concat, y.squeeze(1))
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, evaluator, device):
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

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=12)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--n_layers_1', type=int, default=4)
    parser.add_argument('--n_layers_2', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_first', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = PygNodePropPredDataset('ogbn-papers100M', root='/home2/zwt/')

    split_idx = dataset.get_idx_split()
    data = dataset[0]
    num_feat = data.x.shape[1]
    num_classes = (data.y.data.to(torch.long).max()+1).item()

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.xs = []
            idx = split_idx[split]

            N = data.num_nodes

            t = time.perf_counter()
            print(f'Reading {split} node features...', end=' ', flush=True)

            x = torch.load(f'/home2/zwt/6-yinziqi/gmlp_test/x_{split}_0.pt')
            self.xs.append(x.float())
            for i in range(1, args.num_hops + 1):
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


    model = GMLP(num_feat, args.hidden_channels,
                 num_classes, args.num_hops + 1, args.dropout, 
                 args.input_dropout, args.att_dropout, args.alpha, 
                 args.n_layers_1, args.n_layers_2).to(device)
    '''model = SIGN(num_feat, args.hidden_channels,
                 num_classes, args.num_hops + 1, args.num_layers,
                 args.dropout, not args.no_batch_norm,
                 args.relu_first).to(device)'''
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    evaluator = Evaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.
        valid_test_acc = 0.
        for epoch in range(args.epochs+1):
            t = time.perf_counter()
            train_loss = train(model, train_loader, optimizer, device, epoch, args.epochs)
            if epoch >= 199:
                train_acc = test(model, train_loader, evaluator, device)
                valid_acc = test(model, valid_loader, evaluator, device)
                test_acc = test(model, test_loader, evaluator, device)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(model, './checkpoint/gmlp.pkl')

                logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                if epoch >= 199:
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
