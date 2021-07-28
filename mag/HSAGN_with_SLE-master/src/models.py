import math
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
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

class SIGN(nn.Module):
    def __init__(
        self, in_feats, hidden, out_feats,label_in_feats, num_hops, n_layers, dropout, input_drop, use_labels=False
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        self._use_labels = use_labels
        for i in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, bias=True)

    def forward(self, feats, label_emb):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        if self._use_labels:
            out += self.label_fc(label_emb)
        return out, None

# add batchnorm and replace prelu with relu
class MLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, bias=True, residual=False, relu="relu", batch_norm=True):
        super(MLP, self).__init__()
        self._batch_norm = batch_norm
        self.layers = nn.ModuleList()
        if batch_norm:
            self.bns = nn.ModuleList()
        self.n_layers = n_layers
        self.rec_layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats, bias=bias))
        else:
            self.layers.append(nn.Linear(in_feats, hidden, bias=bias))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden, bias=bias))
                if batch_norm:
                    self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats, bias=bias))
        if self.n_layers > 1:
            self.relu = nn.ReLU() if relu == "relu" else nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        if residual:
            self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_fc = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if isinstance(layer.bias, nn.Parameter):
                nn.init.zeros_(layer.bias)
        if self._batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, x):
        if self.res_fc is not None:
            res_term = self.res_fc(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                if self._batch_norm:
                    x = self.bns[layer_id](x)
                x = self.dropout(self.relu(x))
        if self.res_fc is not None:
            x += res_term
        return x


class SAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, multihop_layers, n_layers, num_heads, relu="relu", batch_norm=True,
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, use_labels=False, use_features=True):
        super(SAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._use_labels = use_labels
        self._use_features = use_features
        self._batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden)
        # self.bns = nn.ModuleList([nn.BatchNorm1d(hidden * num_heads) for i in range(num_hops)])
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        # self.position_emb = nn.Embedding(num_hops, hidden * num_heads)
        self.fcs = nn.ModuleList([MLP(in_feats, hidden, hidden * num_heads, multihop_layers, dropout, relu=relu, batch_norm=batch_norm, bias=True, residual=False) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        # self.res_fc_1 = nn.Linear(hidden, out_feats, bias=False)
        self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, relu=relu, batch_norm=batch_norm,bias=True)

        self.mlp = MLP(hidden, hidden, out_feats, n_layers, dropout, relu=relu, batch_norm=batch_norm, bias=True, residual=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs:
            fc.reset_parameters()
        nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self._use_labels:
            self.label_fc.reset_parameters()
        self.mlp.reset_parameters()
        if self._batch_norm:
            self.bn.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self._use_features:
            feats = [self.input_drop(feat) for feat in feats]
            hidden = []
            for i in range(len(feats)):
                hidden.append(self.fcs[i](feats[i]).view(-1, self._num_heads, self._hidden))
            astack_l = [(feat * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for feat in hidden]
            a_r = (hidden[0] * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.cat([(a_l + a_r).unsqueeze(-1) for a_l in astack_l], dim=-1)
            a = self.leaky_relu(astack)
            a = F.softmax(a, dim=-1)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]
            out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
            out = out.mean(1)
            if self._batch_norm:
                out = self.bn(out)
            out = self.dropout(self.relu(out))
            out = self.mlp(out)
        else:
            a = None
        if self._use_labels:
            out += self.label_fc(label_emb)
        return out, a

class HSAGN(torch.nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, K, n_layers, num_heads, relations_set,
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, last_bias=False, use_labels=False, use_features=True):
        super(HSAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._last_bias = last_bias
        self._use_labels = use_labels
        self._use_features = use_features
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(hidden)
        # self.bns = nn.ModuleList([nn.BatchNorm1d(hidden * num_heads) for i in range(num_hops)])
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        # self.position_emb = nn.Embedding(num_hops, hidden * num_heads)
        self.fcs = nn.ModuleDict({'raw': nn.ModuleList([MLP(in_feats, hidden, hidden * num_heads, n_layers, dropout, bias=True, residual=False)])})
        self.fcs.update(
            {str.join("_", relations): nn.ModuleList([MLP(in_feats, hidden, hidden * num_heads, n_layers, dropout, bias=True, residual=False) for i in range(K)]) for relations in relations_set})
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        # self.res_fc_1 = nn.Linear(hidden, out_feats, bias=False)
        self.hop_attn_l = nn.ParameterDict({'raw': nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))})
        self.hop_attn_l.update(
            {str.join("_", relations): nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden))) for relations in relations_set})
        self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, bias=True)
        if last_bias:
            self.mlp = MLP(hidden, hidden, out_feats, n_layers, dropout, bias=False, residual=False)
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        else:
            self.mlp = MLP(hidden, hidden, out_feats, n_layers, dropout, bias=True, residual=False)
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        for _, fcs in self.fcs.items():
            for fc in fcs:
            #     # nn.init.xavier_normal_(fc.weight, gain=gain)
                fc.reset_parameters()
        # nn.init.xavier_normal_(self.position_emb.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.res_fc_1.weight, gain=gain)
        for _, vec in self.hop_attn_l.items():
            torch.nn.init.xavier_normal_(vec, gain=gain)
        torch.nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self._use_labels:
            self.label_fc.reset_parameters()
        self.mlp.reset_parameters()
        # for bn in self.bns:
        # self.bn.reset_parameters()
        self.bn.reset_parameters()
        if self._last_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, feats_dict, label_emb):
        out = 0
        if self._use_features:
            feats_dict = {str.join("_", relations): [self.input_drop(feat) for feat in feats] \
                          for relations, feats in feats_dict.items()}
            hidden_dict = {}
            for relations, feats in feats_dict.items():
                hidden = []
                start = 0 if relations == "raw" else 1
                for i in range(start, len(feats)):
                    hidden.append(self.fcs[relations][i - start](feats[i]).view(-1, self._num_heads, self._hidden))
                hidden_dict[relations] = hidden

            astack_l_dict = {}
            for relations, hidden in hidden_dict.items():
                astack_l = []
                for i in range(len(hidden)):
                    astack_l.append((hidden[i] * self.hop_attn_l[relations]).sum(dim=-1).unsqueeze(-1))
                astack_l_dict[relations] = astack_l
            a_r = (hidden_dict['raw'][0] * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack_l_flatten = []
            for relations, astack_l in astack_l_dict.items():
                astack_l_flatten += astack_l
            hidden_flatten = []
            for relations, hidden in hidden_dict.items():
                hidden_flatten += hidden
            astack = torch.cat([(a_l + a_r).unsqueeze(-1) for a_l in astack_l_flatten], dim=-1)
            a = self.leaky_relu(astack)
            a = F.softmax(a, dim=-1)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden_flatten[i] * a[:, :, :, i]
            out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
            out = out.mean(1)
            out = self.dropout(self.relu(self.bn(out)))
            out = self.mlp(out)
        else:
            a = None
        if self._use_labels:
            out += self.label_fc(label_emb)
        if self._last_bias:
            out += self.bias
        return out, a

class WeightedAggregator(nn.Module):
    def __init__(self, subset_list, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.num_hops = num_hops
        self.subset_list =subset_list
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(len(subset_list), in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats_dict):
        new_feats = []
        for k in range(self.num_hops):
            feats = torch.cat([feats_dict[rel_subset][k].unsqueeze(1) for rel_subset in self.subset_list], dim=1)
            new_feats.append((feats * self.agg_feats[k].unsqueeze(0)).sum(dim=1))

        return new_feats

class NARS(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, multihop_layers, n_layers, num_heads, subset_list, clf="sagn", relu="relu", batch_norm=True,
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, last_bias=False, use_labels=False, use_features=True):
        super(NARS, self).__init__()
        self.aggregator = WeightedAggregator(subset_list, in_feats, num_hops)
        if clf == "sagn":
            self.clf = SAGN(in_feats, hidden, out_feats, label_in_feats, 
                            num_hops, multihop_layers, n_layers, num_heads, relu=relu, batch_norm=batch_norm,
                            dropout=dropout, input_drop=input_drop, attn_drop=attn_drop, 
                            negative_slope=negative_slope, 
                            use_labels=use_labels, use_features=use_features)
        if clf == "sign":
            self.clf = SIGN(in_feats, hidden, out_feats, label_in_feats,
                            num_hops, n_layers, dropout, input_drop, 
                            use_labels=use_labels)
    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out = self.clf(feats, label_emb)
        return out