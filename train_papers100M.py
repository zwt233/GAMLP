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
import dgl.function as fn
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from logger import Logger
from model import GMLP,GATE,GMLP_SLE
from utils import set_seed
import dgl


def gmlp_train_sle(model, loader, optimizer, evaluator,device, epoch, epochs,xs,labels,label_emb):
    model.train()
    loss_fcn =nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss=0
    num_iter=0
    for idx in loader:
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
        num_iter+=1.0
    loss = total_loss / num_iter
    approx_acc = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    print(f'Epoch:{epoch:.4f}, Loss:{loss:.4f}, Train acc:{approx_acc:.4f}')
    return loss,approx_acc


def gmlp_train_sle_enhance(model, train_loader, enhance_loader, optimizer, evaluator, device, epoch, epochs, xs, labels, label_emb, predict_prob):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        #xs = [torch.from_numpy(x[idx]).float().to(device) for x in xs]
        # print(type(xs[0]))
        # print(xs[0].shape)
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()

        output_att, output_concat = model(feat_list, label_emb[idx].to(device))
        L1 = loss_fcn(output_att[:len(idx_1)],  y.squeeze(1))
        L2 = loss_fcn(output_concat[:len(idx_1)],  y.squeeze(1))
        L3 = F.kl_div(F.log_softmax(output_att[len(idx_1):], dim=1), predict_prob[idx_2].to(device), reduction='batchmean')
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2 +   L3
        loss.backward()
        optimizer.step()

        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss

    loss = total_loss / len(train_loader.dataset)
    approx_acc = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    #print(f'Epoch:{epoch:.4f}, Loss:{loss:.4f}, Train acc:{approx_acc:.4f}')
    return loss, approx_acc


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
def neighbor_average_features(g, feat):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata['f'] = feat
    g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
    feat = g.ndata.pop('f')
    gc.collect()
    return feat
def prepare_label_emb(args,label_teacher_emb=None):
    from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
    dataset = DglNodePropPredDataset(name='ogbn-papers100M', root='/data4/zwt/')
    splitted_idx = dataset.get_idx_split()
    train_idx = splitted_idx["train"]
    valid_idx = splitted_idx["valid"]
    test_idx = splitted_idx["test"]
    g, labels = dataset[0]
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    n_classes = dataset.num_classes
    del g.ndata['feat']
    del dataset
    print(n_classes)
    if label_teacher_emb==None:
        y = np.zeros(shape=(labels.shape[0],int(n_classes)))
        y[train_idx]=F.one_hot(labels[train_idx].to(torch.long),num_classes=n_classes).float().squeeze(1)
        y=torch.Tensor(y)
    else:
        y = np.zeros(shape=(labels.shape[0],int(n_classes)))
        y[valid_idx]=label_teacher_emb[len(train_idx):len(train_idx)+len(valid_idx)]
        y[test_idx]=label_teacher_emb[len(train_idx)+len(valid_idx):len(train_idx)+len(valid_idx)+len(test_idx)]
        y[train_idx]=F.one_hot(labels[train_idx].to(torch.long),num_classes=n_classes).float().squeeze(1)
        y=torch.Tensor(y)
    del labels
    del label_teacher_emb
    for hop in range(args.label_num_hops):
        y = neighbor_average_features(g, y)
        gc.collect()
    return torch.cat([y[train_idx],y[valid_idx],y[test_idx]],dim=0)
def prepare_data(args,stage,label_teacher_emb):
    label_emb=prepare_label_emb(args,label_teacher_emb)
    gc.collect()
    print("load label embedding is over")
    dataset = PygNodePropPredDataset('ogbn-papers100M', root='/data4/zwt/')
    torch.set_default_tensor_type(torch.FloatTensor)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    num_feat = data.x.shape[1]
    num_classes = int((data.y.data.to(torch.long).max()+1).item())
    train_nid = split_idx['train']
    val_nid = split_idx['valid']
    test_nid = split_idx['test']
    labels=data.y.data
    labels=torch.cat([labels[train_nid],labels[val_nid],labels[test_nid]])
    xs=[]
    for i in range(args.num_hops+1):
        xs.append(torch.load(f'/data4/zwt/ogbn_papers100M/feat/papers100m_feat_{i}.pt'))
    evaluator = Evaluator(name='ogbn-papers100M')
    gc.collect()
    '''train_loader = torch.utils.data.DataLoader(
        torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        len(train_nid)+torch.arange(len(val_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        len(train_nid)+len(val_nid)+torch.arange(len(test_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    all_loader = torch.utils.data.DataLoader(
        torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
        shuffle=False, drop_last=False)'''

    return xs,label_emb,train_nid,val_nid,test_nid,labels,evaluator,num_feat,num_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=12)
    parser.add_argument('--label_num_hops',type=int,default=9)
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
    parser.add_argument('--warm_start', type=int, default=0)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=30000)
    parser.add_argument('--stages', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=300)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    #xs,label_emb,train_nid,val_nid,test_nid,labels,evaluator,num_feat,num_classes=prepare_data(args,stage,)
    import uuid
    checkpt_file = './output/'+uuid.uuid4().hex
    for run in range(args.runs):
        set_seed(run)
        #logger = Logger(args.stages, args)

        if run == 0:
            st = args.warm_start
        else:
            st = 0

        for stage in range(st, args.stages):
            if stage > 0:
                ensemble_prob = torch.load(checkpt_file+f'_{stage-1}.pt')
                train_node_nums = len(train_nid)
                valid_node_nums = len(val_nid)
                test_node_nums = len(test_nid)
                total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

                print("This model Train ACC is {}".format(evaluator.eval({
                    'y_true': labels[:train_node_nums],
                    'y_pred': ensemble_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))

                print("This model Valid ACC is {}".format(evaluator.eval({
                    'y_true': labels[train_node_nums:train_node_nums+valid_node_nums],
                    'y_pred': ensemble_prob[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))

                print("This model Test ACC is {}".format(evaluator.eval({
                    'y_true': labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],
                    'y_pred': ensemble_prob[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))

                tr_va_te_nid = torch.arange(total_num_nodes)
                confident_nid = torch.arange(len(ensemble_prob))[
                    ensemble_prob.max(1)[0] > args.threshold]
                extra_confident_nid = confident_nid[confident_nid >= len(
                    train_nid)]
                print(f'Stage: {stage}, confident nodes: {len(extra_confident_nid)}')

                real_idx = torch.cat((train_nid, val_nid, test_nid), dim=0)
                teacher_probs = torch.zeros(ensemble_prob.shape[0], ensemble_prob.shape[1])
                teacher_probs[extra_confident_nid,:] = ensemble_prob[extra_confident_nid,:]

                enhance_idx = extra_confident_nid
                xs,label_emb,train_nid,val_nid,test_nid,labels,evaluator,num_feat,num_classes=prepare_data(args,stage,teacher_probs)

                if len(extra_confident_nid) > 0:
                    enhance_loader = torch.utils.data.DataLoader(enhance_idx, batch_size=int(args.batch_size*len(enhance_idx)/(len(train_nid)+len(enhance_idx))), shuffle=True, drop_last=False)
                    train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=int(args.batch_size*len(train_nid)/(len(train_nid)+len(enhance_idx))), shuffle=True, drop_last=False)
            else:
                teacher_probs=None
                xs,label_emb,train_nid,val_nid,test_nid,labels,evaluator,num_feat,num_classes=prepare_data(args,stage,None)
                train_loader = torch.utils.data.DataLoader(
                            torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
            valid_loader = torch.utils.data.DataLoader(
                            len(train_nid)+torch.arange(len(val_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader = torch.utils.data.DataLoader(
                            len(train_nid)+len(val_nid)+torch.arange(len(test_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)
            all_loader = torch.utils.data.DataLoader(
                            torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)

            model = GMLP_SLE(num_feat,num_classes, args.hidden_channels,
                    num_classes, args.num_hops+1, args.dropout,
                    args.input_dropout, args.att_dropout,0, args.alpha,
                    args.n_layers_1, args.n_layers_2,args.n_layers_3,args.pre_process).to(device)
            num_params = sum([p.numel() for p in model.parameters()])
            print(f'#Params: {num_params}')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_acc = 0.
            valid_test_acc = 0.
            for epoch in range(args.epochs+1):
                t = time.perf_counter()
                if stage > 0 and len(extra_confident_nid) > 0:
                    train_loss,train_acc = gmlp_train_sle_enhance(model, train_loader, enhance_loader, optimizer,evaluator, device, epoch, args.epochs,xs, labels, label_emb, ensemble_prob)
                else:
                    train_loss,train_acc = gmlp_train_sle(model, train_loader, optimizer,evaluator, device, epoch, args.epochs,
                        xs, labels, label_emb)
                valid_acc = gmlp_test_sle(model, valid_loader, evaluator, device,xs,labels,label_emb)
                test_acc = gmlp_test_sle(model, test_loader, evaluator, device,xs,labels,label_emb)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    valid_test_acc = test_acc
                    torch.save(model, checkpt_file+f'_{stage}.pkl')

                #logger.add_result(stage, (train_acc, valid_acc, test_acc))

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                        f'Stage: {stage + 1:02d}, '
                        f'Epoch: {epoch + 1:02d}, '
                        f'Training loss: {train_loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}%, '
                        f'Test: {100 * test_acc:.2f}%, '
                        f'Best: {100 * valid_test_acc:.2f}%, '
                        f'Time: {time.perf_counter()-t:.4f}')

            model = torch.load(checkpt_file+f'_{stage}.pkl')
            teacher_probs = gmlp_gen_output_sle(model,all_loader,evaluator,device,xs,labels,label_emb)
            torch.save(teacher_probs, checkpt_file+f'_{stage}.pt')

            #logger.print_statistics(stage)
        #logger.print_statistics()

        ensemble_prob = torch.load(checkpt_file+f'_0.pt')
        for i in range(1, args.stages):
            ensemble_prob = ensemble_prob + torch.load(checkpt_file+f'_{i}.pt')
        ensemble_prob /= args.stages

        print("This model Train ACC is {}".format(evaluator.eval({
            'y_true': labels[:train_node_nums],
            'y_pred': ensemble_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu()
        })['acc']))

        print("This model Valid ACC is {}".format(evaluator.eval({
            'y_true': labels[train_node_nums:train_node_nums+valid_node_nums],
            'y_pred': ensemble_prob[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu()
        })['acc']))

        print("This model Test ACC is {}".format(evaluator.eval({
            'y_true': labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],
            'y_pred': ensemble_prob[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu()
        })['acc']))
