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
import gc

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from logger import Logger
from model import GMLP,GATE,GMLP_SLE
from utils import set_seed

def gmlp_train_sle(model, loader, optimizer, evaluator,device, epoch, epochs,xs,labels,label_emb):
    model.train()
    loss_fcn =nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss=0
    for idx in loader:
        #xs = [torch.from_numpy(x[idx]).float().to(device) for x in xs]
        #print(type(xs[0]))
        #print(xs[0].shape)
        feat_list = [x[idx].to(device) for x in xs]        
        y = labels[idx].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att,output_concat = model(feat_list,label_emb[idx].to(device))
        L1 = loss_fcn(output_att,  y.squeeze(1))
        L2 = loss_fcn(output_concat,  y.squeeze(1))
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2    
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss+=loss
    loss = total_loss / len(loader.dataset)        
    approx_acc = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    print(f'Epoch:{epoch:.4f}, Loss:{loss:.4f}, Train acc:{approx_acc:.4f}')
    return loss,approx_acc
    
@torch.no_grad()
def gmlp_test_sle(model, loader, evaluator, device, xs,labels,label_emb):
    model.eval()
    y_true, y_pred = [], []
    for idx in loader:
        feat_list = [x[idx].to(device) for x in xs]   
        y_true.append(labels[idx].to(torch.long))
        output = model(feat_list,label_emb[idx].to(device))[0]
        y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())
    return evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
@torch.no_grad()    
def gmlp_gen_output_sle(model,loader,evaluator,device,xs,labels,label_emb):
    model.eval()
    output_list = []    
    for idx in loader:
        feat_list = [x[idx].to(device) for x in xs] 
        output = model(feat_list,label_emb[idx].to(device))[0].softmax(dim=1).cpu()
        output_list.append(output)
    return torch.cat(output_list,dim=0)
    
    
def gmlp_train(model, loader, optimizer, evaluator,device, epoch, epochs):
    model.train()
    total_loss = total_correct = total_node=0
    loss_fcn =nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    for xs, y,label_emb,idx in loader:
#        print(label_emb.shape)
#        print(xs[0].shape)
#        print(y.shape)
        xs = [x.to(device) for x in xs]
        y = y.to(torch.long).to(device)
        optimizer.zero_grad()
        output_att,output_concat = model(xs,label_emb.to(device))
        L1 = loss_fcn(output_att,  y.squeeze(1))
        L2 = loss_fcn(output_concat,  y.squeeze(1))
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2
        #total_loss += float(loss)
        y_true.append(y.to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())        
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * y.numel()
    
    loss = total_loss / len(loader)
    approx_acc = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']

    print(f'Epoch:{epoch:.4f}, Loss:{loss:.4f}, Train acc:{approx_acc:.4f}')
    
    return loss,approx_acc

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
    for xs, y,label_emb,idx in loader:
        xs = [x.to(device) for x in xs]
        y_true.append(y.to(torch.long))
        output = model(xs,label_emb.to(device))[0]
        y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())

    return evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    
def gmlp_gen_output(model,loader,evaluator,device,probs):
    model.eval()
    y_true, y_pred = [], []    
    for xs, y,label_emb,idx in loader:
        xs = [x.to(device) for x in xs]
        y_true.append(y.to(torch.long))
        output = model(xs,label_emb.to(device))[0]
        y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())
        probs[idx]=output.softmax(dim=1)
    print('acc:{:.4f}').format(evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc'])
    return probs
    

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
    parser.add_argument('--device', type=int, default=1)
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
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.7)    
    parser.add_argument('--stage', type=list, default=[300,300,300])      
    
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = PygNodePropPredDataset('ogbn-papers100M', root='../../')
    torch.set_default_tensor_type(torch.FloatTensor)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    num_nodes=data.num_nodes
    num_feat = data.x.shape[1]
    num_classes = int((data.y.data.to(torch.long).max()+1).item())
    train_nid = split_idx['train']
    val_nid = split_idx['valid']
    test_nid = split_idx['test']
    labels=data.y.data
    
    #print(len(train_nid))
    #print(torch.arange(len(train_nid)))
    train_loader = torch.utils.data.DataLoader(
        torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        len(train_nid)+torch.arange(len(val_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        len(train_nid)+len(val_nid)+torch.arange(len(test_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    all_loader = torch.utils.data.DataLoader(
        torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    train_node_nums=len(train_nid)
    valid_node_nums=len(val_nid)
    test_node_nums=len(test_nid)        
    total_num_nodes = train_node_nums + valid_node_nums + test_node_nums
    #print(total_num_nodes)
    '''
    mapping_array=np.zeros((num_nodes,1))
    mapping_array[train_nid]=torch.arrange(train_node_nums)
    mapping_array[val_nid]=torch.arrange(valid_node_nums)+train_node_nums
    mapping_array[test_nid]=torch.arrange(test_node_nums)+train_node_nums+valid_node_nums
    '''    
    labels=torch.cat([labels[train_nid],labels[val_nid],labels[test_nid]],dim=0)
    label_emb_train = torch.load(f'./dgl_y_train.pt')
    label_emb_val = torch.load(f'./dgl_y_valid.pt')
    label_emb_test = torch.load(f'./dgl_y_test.pt')
    label_emb = torch.zeros(size=(total_num_nodes,label_emb_train.shape[1]))
    
    label_emb[:train_node_nums,:]=label_emb_train.float()
    label_emb[train_node_nums:train_node_nums+valid_node_nums,:]=label_emb_val.float()
    label_emb[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums,:]=label_emb_test.float()     
    
    del label_emb_train
    del label_emb_val            
    del label_emb_test    
    gc.collect()
    xs = []    
    
    for i in range(args.num_hops + 1):
        xs.append(torch.zeros(size=(total_num_nodes,data.x.shape[1])))    
    for i in range(args.num_hops + 1):
        xs[i][:train_node_nums,:]= torch.load(f'./x_train_{i}.pt')
        xs[i][train_node_nums:train_node_nums+valid_node_nums,:]= torch.load(f'./x_valid_{i}.pt')
        xs[i][train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums,:]= torch.load(f'./x_test_{i}.pt')                 
        gc.collect()
    evaluator = Evaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        set_seed(run)    
        model = GMLP_SLE(num_feat,num_classes, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.n_layers_3,args.pre_process).to(device)
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.
        valid_test_acc = 0.    
        for epoch in range(args.stage[0]+1):
          t = time.perf_counter()
          train_loss,train_acc = gmlp_train_sle(model, train_loader, optimizer,evaluator, device, epoch, args.stage[0],xs,labels,label_emb)
          if epoch >= 49:
                #train_acc = gmlp_train_sle(model, train_loader, evaluator, device)
                valid_acc = gmlp_test_sle(model, valid_loader, evaluator, device,xs,labels,label_emb)
                test_acc = gmlp_test_sle(model, test_loader, evaluator, device,xs,labels,label_emb)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(model, f'./checkpoint/gmlp_sle_{args.num_hops}_{run}.pkl')

                logger.add_result(run, (train_acc, valid_acc, test_acc))

          if epoch % args.log_steps == 0:
             if epoch >= 49:
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
                          f'Train: {100 * train_acc:.2f}%, '                          
                          f'Time: {time.perf_counter()-t:.4f}')
        model = torch.load(f'./checkpoint/gmlp_sle_{args.num_hops}_{run}.pkl')
        #model.load_state_dict(state_dict)
        
        teacher_probs=gmlp_gen_output_sle(model,all_loader,evaluator,device,xs,labels,label_emb)
        
        print("This model Train ACC is {}".format(evaluator.eval({
          'y_true': labels[:train_node_nums],
          'y_pred': teacher_probs[:train_node_nums].argmax(dim=-1, keepdim=True).cpu()
      })['acc']))              
      
          
        print("This model Valid ACC is {}".format(evaluator.eval({
          'y_true': labels[train_node_nums:train_node_nums+valid_node_nums],
          'y_pred': teacher_probs[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu()
      })['acc']))        
        best_acc=evaluator.eval({
          'y_true': labels[train_node_nums:train_node_nums+valid_node_nums],
          'y_pred': teacher_probs[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu()
      })['acc']
      
        print("This model Test ACC is {}".format(evaluator.eval({
          'y_true': labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],
          'y_pred': teacher_probs[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu()
      })['acc']))       
        continue
        #exit()
        tr_va_te_nid = torch.arange(total_num_nodes)          
        confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > args.threshold]        
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(train_nid)]
        confident_nid = tr_va_te_nid[confident_nid_inner]        
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]                    
        print(f"pseudo label number: {len(confident_nid)}")
        pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
        labels_with_pseudos = labels.clone().to(torch.long)
        train_nid_with_pseudos = np.union1d(torch.arange(train_node_nums), extra_confident_nid_inner)
        print(f"enhanced train set number: {len(train_nid_with_pseudos)}")
        #labels_with_pseudos[:len(train_nid)] = labels[:len(train_nid)]
        
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid_inner].reshape(-1,1).to(torch.long)
        pseudo_train_loader = torch.utils.data.DataLoader(
        train_nid_with_pseudos, batch_size=args.batch_size, shuffle=True, drop_last=False)          
        for epoch in range(args.stage[1]+1):
            train_loss,train_acc = gmlp_train_sle(model, pseudo_train_loader, optimizer,evaluator, device, epoch, args.stage[1],xs,labels_with_pseudos,label_emb)                
            train_acc = gmlp_test_sle(model, train_loader, evaluator, device,xs,labels,label_emb)
            valid_acc = gmlp_test_sle(model, valid_loader, evaluator, device,xs,labels,label_emb)
            test_acc = gmlp_test_sle(model, test_loader, evaluator, device,xs,labels,label_emb)
            logger.add_result(run, (train_acc, valid_acc, test_acc))
            print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch + 1:02d}, '
                          f'Training loss: {train_loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%, '
                          f'Best: {100 * valid_test_acc:.2f}%, '
                          f'Time: {time.perf_counter()-t:.4f}')               
            if valid_acc > best_acc:
                best_acc = valid_acc
                valid_test_acc = test_acc
                if not os.path.isdir('checkpoint'):
                  os.mkdir('checkpoint')
                  torch.save(model, './checkpoint/enhenced_gmlp_sle_{args.num_hops}_{run}.pkl')
            
    exit()        
    '''
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.xs = []
            idx = split_idx[split]

            N = data.num_nodes

            t = time.perf_counter()
            print(f'Reading {split} node features...', end=' ', flush=True)

            x = data.x[idx]
            #self.xs.append(x.float())
            self.label_emb = torch.load(f'./dgl_y_{split}.pt')
            for i in range(args.num_hops + 1):
                x = torch.load(f'./x_{split}_{i}.pt')
                self.xs.append(x.float())
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            self.y = (data.y.data)[idx].to(torch.long)

        def __len__(self):
            return self.xs[0].size(0)

        def __getitem__(self, idx):
            return [x[idx] for x in self.xs], self.y[idx],self.label_emb[idx],idx
    
    
    
    class AllMyDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.xs = []
            idx =torch.cat([split_idx['train'],split_idx['val'],split_idx['test']])
            N = data.num_nodes
            t = time.perf_counter()
            print(f'Reading {split} node features...', end=' ', flush=True)

            x = data.x[idx]
            #self.xs.append(x.float())
            label_emb_train = torch.load(f'./dgl_y_train.pt')
            label_emb_val = torch.load(f'./dgl_y_valid.pt')
            label_emb_test = torch.load(f'./dgl_y_test.pt')
            
            self.label_emd = np.zeros(shape=(data.num_nodes,label_emb_train.shape[1]))
            self.label_emd[split_idx['train']]=label_emb_train
            self.label_emd[split_idx['valid']]=label_emb_val
            self.label_emd[split_idx['test']]=label_emb_test          
            del label_emb_train
            del label_emb_val            
            del label_emb_test
            gc.collect()
            self.label_emd = torch.Tensor(self.label_emd)
            for i in range(args.num_hops + 1):
                self.xs.append(np.zeros(shape=(data.num_nodes,data.x.shape[1])))
            for split in ['train','valid','test']:
                for i in range(args.num_hops + 1):
                    x = torch.load(f'./x_{split}_{i}.pt')
                    self.xs[i][split_idx[split]]=x.float()
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
            self.y = data.y.data.to(torch.long)

        def __len__(self):
            return self.xs[0].size(0)

        def __getitem__(self, idx):
            return [x[idx] for x in self.xs], self.y[idx],self.label_emb[idx],idx
            
    
    train_dataset = MyDataset(split='train')
    valid_dataset = MyDataset(split='valid')
    test_dataset = MyDataset(split='test')

    #train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
    #                          num_workers=6, persistent_workers=True)
    #valid_loader = DataLoader(valid_dataset, args.batch_size)
    #test_loader = DataLoader(test_dataset, args.batch_size)  
    all_dataset = AllMyDataset()
    '''
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
        '''
        model = GMLP(num_feat, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.pre_process).to(device)  
        '''                 
        model = GMLP_SLE(num_feat,num_classes, args.hidden_channels,
                 num_classes, args.num_hops+1, args.dropout,
                 args.input_dropout, args.att_dropout, args.alpha,
                 args.n_layers_1, args.n_layers_2,args.n_layers_3,args.pre_process).to(device)
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.
        valid_test_acc = 0.
        for epoch in range(args.epochs+1):
            t = time.perf_counter()
            train_loss,train_acc = gmlp_train(model, train_loader, optimizer,evaluator, device, epoch, args.epochs)
            if epoch >= 49:
                #train_acc = gmlp_test(model, train_loader, evaluator, device)
                valid_acc = gmlp_test(model, valid_loader, evaluator, device)
                test_acc = gmlp_test(model, test_loader, evaluator, device)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(model, './checkpoint/gmlp_sle.pkl')

                logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                if epoch >= 49:
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
                          f'Train: {100 * train_acc:.2f}%, '                          
                          f'Time: {time.perf_counter()-t:.4f}')
        model.load_state_dic(torch.load('./checkpoint/gmlp_sle2.pkl'))
        teacher_probs=np.zeros(shape=(data.x.shape[0],num_classes))
        with torch.no_grad():
              best_model.eval()   
              teacher_probs=gmlp_gen_output(model,train_loader,evaluator,device,teacher_probs)
              gc.collect()  
              teacher_probs=gmlp_gen_output(model,valid_loader,evaluator,device,teacher_probs)
              gc.collect()  
              teacher_probs=gmlp_gen_output(model,test_loader,evaluator,device,teacher_probs)  
              gc.collect()  
        confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > args.threshold]
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(split_idx['train'])]
        confident_nid = tr_va_te_nid[confident_nid_inner]
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]
        print(f"pseudo label number: {len(confident_nid)}")
        pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
        for epoch in range(args.epochs+1):
                train_loss,train_acc = sle_gmlp_train(model, new_train_loader, optimizer,evaluator, device, epoch, args.epochs)                
                train_acc = gmlp_test(model, train_loader, evaluator, device)
                valid_acc = gmlp_test(model, valid_loader, evaluator, device)
                test_acc = gmlp_test(model, test_loader, evaluator, device)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(model, './checkpoint/gmlp_sle.pkl')

                logger.add_result(run, (train_acc, valid_acc, test_acc))
                print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch + 1:02d}, '
                          f'Training loss: {train_loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%, '
                          f'Best: {100 * valid_test_acc:.2f}%, '
                          f'Time: {time.perf_counter()-t:.4f}')                
                          
        logger.print_statistics(run)
    logger.print_statistics()

