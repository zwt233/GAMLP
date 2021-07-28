from layer import *
        
class GATE(nn.Module):
    def __init__(self, nfeat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,pre_process=False):
        super(GATE, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        if pre_process:
          self.lr_att = torch.Tensor(size=(hidden, 1))      
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)        
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])      
        else:
          self.lr_att = nn.Linear(size=(nfeat, 1))      
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)           
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lr_att, gain=nn.init.calculate_gain('relu'))
        self.lr_right1.reset_parameters()
        if self.pre_process:
            for layer in self.fcs:
                layer.reset_parameters()
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
          self.lr_att = torch.Tensor(size=(hidden, 1))
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)        
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])      
        else:
          self.lr_att = torch.Tensor(size=(hidden, 1))
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)           
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lr_att, gain=nn.init.calculate_gain('relu'))
        self.lr_right1.reset_parameters()
        if self.pre_process:
            for layer in self.fcs:
                layer.reset_parameters() 
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
          self.lr_att = nn.Linear(hidden+hidden, 1)    
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)        
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])      
        else:
          self.lr_left1 = FeedForwardNetII(num_hops*nfeat, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(hidden+nfeat, 1)      
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)           
        self.lr_left2 = nn.Linear(hidden, nclass)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
#        self.res_fc=nn.Linear(nfeat,hidden,bias=False)
#        self.norm=nn.BatchNorm1d(hidden)
        self.reset_parameters()
        self.act=torch.nn. LeakyReLU()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lr_att.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.lr_left2.weight, gain=nn.init.calculate_gain('relu'))       
        self.lr_left1.reset_parameters()
        self.lr_right1.reset_parameters()
        if self.pre_process:
            for layer in self.fcs:
                layer.reset_parameters()

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
        #right_1=self.dropout(self.prelu(right_1))
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
        self.lr_left2 = nn.Linear(hidden, nclass) 
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.label_fc= FeedForwardNet(label_feat, hidden, nclass, n_layers_3, dropout)     
        self.norm=nn.BatchNorm1d(hidden)        
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
        #right_1+=self.res_fc(feature_list[0])
        right_1 = self.lr_right1(self.prelu(self.dropout(right_1)))
        right_1 += self.label_fc(label_emb)         
        return right_1,left_2
