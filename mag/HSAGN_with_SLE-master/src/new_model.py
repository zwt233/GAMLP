from new_layer import *

class GATE(nn.Module):
    def __init__(self, nfeat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,pre_process=False):
        super(GATE, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        if pre_process:
          self.lr_att = nn.Linear(hidden, 1)
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        else:
          self.lr_att = nn.Linear(nfeat, 1)
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
    def forward(self, feature_list):
        num_node=feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        if self.pre_process:
            for i in range(len(feature_list)):
                feature_list[i]=self.fcs[i](feature_list[i])
        attention_scores = [torch.sigmoid(self.lr_att(x).view(num_node, 1)) for x in feature_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(feature_list[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(feature_list[i],self.att_drop(W[:, i].view(num_node, 1)))
        right_1 = self.lr_right1(right_1)
        return right_1


class R_GATE(nn.Module):
    def __init__(self, nfeat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,num_node,pre_process=False):
        super(R_GATE, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        if pre_process:
          self.lr_att = nn.Linear(hidden + hidden, 1)
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        else:
          self.lr_att = nn.Linear(nfeat + nfeat, 1)
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
    def forward(self, feature_list):
        num_node=feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        if self.pre_process:
            for i in range(self.num_hops):
                feature_list[i]=self.fcs[i](feature_list[i])
        attention_scores=[]
        attention_scores.append(torch.sigmoid(self.lr_att(torch.cat([feature_list[0],feature_list[0]],dim=1))))
        for i in range(1,self.num_hops):
            history_att=torch.cat(attention_scores[:i],dim=1)
            att=F.softmax(history_att, 1)
            history = torch.mul(feature_list[0], self.att_drop(att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history=history+torch.mul(feature_list[j],self.att_drop(att[:, j].view(num_node, 1)))
            attention_scores.append(torch.sigmoid(self.lr_att(torch.cat([history,feature_list[i]],dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(feature_list[0], self.att_drop(attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(feature_list[i],self.att_drop(attention_scores[:, i].view(num_node, 1)))
        right_1 = self.lr_right1(right_1)
        return right_1
class GMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,pre_process=False):
        super(GMLP, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        if pre_process:
          self.lr_left1 = FeedForwardNetII(num_hops*hidden, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(hidden + hidden, 1)
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        else:
          self.lr_left1 = FeedForwardNetII(num_hops*nfeat, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(nfeat + hidden, 1)
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)
        self.lr_left2 = nn.Linear(hidden, nclass)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.res_fc=nn.Linear(nfeat,hidden,bias=False)
        self.norm=nn.BatchNorm1d(hidden)
    def forward(self, feature_list):
        num_node=feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        hidden_list=[]
        if self.pre_process:
            for i in range(len(feature_list)):
                hidden_list.append(self.fcs[i](feature_list[i]))
        concat_features = torch.cat(hidden_list, dim=1)
        left_1 =self.dropout(self.prelu(self.lr_left1(concat_features)))
        left_2 = self.lr_left2(left_1)
        attention_scores = [torch.sigmoid(self.lr_att(torch.cat((left_1, x), dim=1))).view(num_node, 1) for x in
                            hidden_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(hidden_list[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(hidden_list[i],self.att_drop(W[:, i].view(num_node, 1)))
        #right_1 += self.res_fc(feature_list[0])
        #right_1= self.norm(right_1)
        right_1=self.dropout(self.prelu(right_1))
        right_1 = self.lr_right1(right_1)
        return right_1,left_2



class GMLP_SLE(nn.Module):
    def __init__(self, nfeat,label_feat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,n_layers_3,pre_process=False):
        super(GMLP_SLE, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        self.res_fc=nn.Linear(nfeat , hidden,bias=False)
        if pre_process:
          self.lr_left1 = FeedForwardNetII(num_hops*hidden, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(hidden + hidden, 1)
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        else:
          self.lr_left1 = FeedForwardNetII(num_hops*nfeat, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(nfeat + hidden, 1)
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)
        self.res_fc=FeedForwardNet(nfeat,hidden,hidden,2,dropout)
        self.lr_left2 = nn.Linear(hidden, nclass)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.label_fc= FeedForwardNet(label_feat, hidden, nclass, n_layers_3, dropout)
        self.norm=nn.BatchNorm1d(hidden)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lr_att.weight, gain=nn.init.calculate_gain('relu'))
        self.lr_right1.reset_parameters()
        self.res_fc.reset_parameters()
        self.label_fc.reset_parameters()
        if self.pre_process:
            for layer in self.fcs:
                layer.reset_parameters()
    def forward(self, feature_list,label_emb):
        num_node=feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        hidden_list=[]
        if self.pre_process:
            for i in range(len(feature_list)):
                hidden_list.append(self.fcs[i](feature_list[i]))
        concat_features = torch.cat(hidden_list, dim=1)
        left_1 =self.dropout(self.prelu(self.lr_left1(concat_features)))
        left_2 = self.lr_left2(left_1)
        attention_scores = [torch.sigmoid(self.lr_att(torch.cat((left_1, x), dim=1))).view(num_node, 1) for x in
                            hidden_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(hidden_list[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(hidden_list[i],self.att_drop(W[:, i].view(num_node, 1)))
        right_1+=self.res_fc(feature_list[0])
        right_1 = self.lr_right1(self.prelu(self.dropout(right_1)))
        #right_1 = self.lr_right1(right_1)
        right_1 += self.label_fc(label_emb)
        return right_1,left_2
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
'''
class R_GATE(nn.Module):
        def __init__(self, nfeat, hidden, nclass,num_hops,
                                 dropout, input_drop,att_dropout,n_layers_1,n_layers_2,use_labels,label_feat):
            super(R_GATE, self).__init__()
            self.num_hops=num_hops
            self.prelu=nn.PReLU()
            self.lr_att = nn.Linear(hidden + hidden, 1)
           `self.re_att = nn.Linear(hidden + hidden, 1)
            self.fcs = nn.ModuleList([MLP(nfeat, hidden, hidden , n_layers_1, dropout) for i in range(num_hops)])
            self.lr_right1 = MLP(hidden, hidden, nclass, n_layers_2, dropout)
            self.res_fc = nn.Linear(nfeat, hidden , bias=False)
            self.dropout = nn.Dropout(dropout)
            self.input_drop = nn.Dropout(input_drop)
                                                                                         self.att_drop=nn.Dropout(att_dropout)
                                                                                                            self._use_labels=use_labels
                                                                                                                    self.label_fc = MLP(label_feat, hidden, nclass, 2 * n_layers_1, dropout, bias=True)
                                                                                                                            self.leaky_relu = nn.LeakyReLU(0.2)
                                                                                                                                    self.bn = nn.BatchNorm1d(hidden)
                                                                                                                                            self.relu = nn.ReLU()
                                                                                                                                                def forward(self, feature_list,label_embed):
                                                                                                                                                            num_node=feature_list[0].shape[0]
                                                                                                                                                                    feature_list = [self.input_drop(feature) for feature in feature_list]
                                                                                                                                                                            hidden_list=[]
                                                                                                                                                                                    for i in range(self.num_hops):
                                                                                                                                                                                                    hidden_list.append(self.fcs[i](feature_list[i]))
                                                                                                                                                                                                            attention_scores=[]
                                                                                                                                                                                                                    attention_scores.append(self.lr_att(torch.cat([hidden_list[0],hidden_list[0]],dim=1)))
                                                                                                                                                                                                                            for i in range(1,self.num_hops):
                                                                                                                                                                                                                                            history_att=self.leaky_relu(torch.cat(attention_scores[:i],dim=1))
                                                                                                                                                                                                                                                        att=F.softmax(history_att, 1)
                                                                                                                                                                                                                                                        #            history=hidden_list[0]
                                                                                                                                                                                                                                                                    history = torch.mul(hidden_list[0],att[:, 0].view(num_node, 1))
                                                                                                                                                                                                                                                                                for j in range(1, i):
                                                                                                                                                                                                                                                                                                    history=history+torch.mul(hidden_list[j],self.att_drop(att[:, j].view(num_node, 1)))
                                                                                                                                                                                                                                                                                                                attention_scores.append((self.lr_att(torch.cat([history,hidden_list[i]],dim=1))))
                                                                                                                                                                                                                                                                                                                        attention_scores = self.leaky_relu(torch.cat(

class NARS_rgate(nn.Module):
       def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, multihop_layers, n_layers, num_heads, subset_list, clf="sagn", relu="relu", batch_norm=True,dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, last_bias=False, use_labels=False, use_features=True):
           super(NARS_gmlp, self).__init__()
           self.aggregator = WeightedAggregator(subset_list, in_feats, num_hops)
           if clf == "sagn":
                self.clf = SAGN(in_feats, hidden, out_feats, label_in_feats,num_hops, multihop_layers, n_layers, num_heads, relu=relu, batch_norm=batch_norm, dropout=dropout, input_drop=input_drop, attn_drop=attn_drop,negative_slope=negative_slope,use_labels=use_labels, use_features=use_features)
           if clf == "sign":
                self.clf = SIGN(in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, dropout, input_drop,use_labels=use_labels)
           if clf == "gmlp":
                self.clf=GMLP_SLE(in_feats,label_in_feats, hidden, out_feats,num_hops,dropout, input_drop,attn_drop,0.5,2,2,4,pre_process=True)
           if clf=="rgate"
                self.clf=rgate_sle()
       def forward(self, feats_dict, label_emb):
           feats = self.aggregator(feats_dict)
           out1,out2 = self.clf(feats, label_emb)
           return out1,out2
'''
class NARS_gmlp(nn.Module):
       def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, multihop_layers, n_layers, num_heads, subset_list, clf="sagn", relu="relu", batch_norm=True,dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, last_bias=False, use_labels=False, use_features=True):
           super(NARS_gmlp, self).__init__()
           self.aggregator = WeightedAggregator(subset_list, in_feats, num_hops)
           if clf == "sagn":
                self.clf = SAGN(in_feats, hidden, out_feats, label_in_feats,num_hops, multihop_layers, n_layers, num_heads, relu=relu, batch_norm=batch_norm, dropout=dropout, input_drop=input_drop, attn_drop=attn_drop,negative_slope=negative_slope,use_labels=use_labels, use_features=use_features)
           if clf == "sign":
                self.clf = SIGN(in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, dropout, input_drop,use_labels=use_labels)
           if clf == "gmlp":
                self.clf=GMLP_SLE(in_feats,label_in_feats, hidden, out_feats,num_hops,dropout, input_drop,attn_drop,0.5,4,6,4,pre_process=True)
       def forward(self, feats_dict, label_emb):
           feats = self.aggregator(feats_dict)
           out1,out2 = self.clf(feats, label_emb)
           return out1,out2
