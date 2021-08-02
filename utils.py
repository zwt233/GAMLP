import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from load_dataset import load_dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
from model import R_GAMLP,JK_GAMLP,NARS_JK_GAMLP,NARS_R_GAMLP,R_GAMLP_RDD,JK_GAMLP_RDD,NARS_JK_GAMLP_RDD,NARS_R_GAMLP_RDD

def gen_model_mag(args,num_feats,in_feats,num_classes):
    if args.method=="R_GAMLP":
        return NARS_R_GAMLP(in_feats, hidden, num_classes, args.num_hops+1,num_classes,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual)
    elif args.method=="JK_GAMLP":
        return NARS_JK_GAMLP(in_feats, hidden, num_classes, args.num_hops+1,num_classes,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual)

def gen_model_mag_rdd(args,num_feats,in_feats,num_classes):
    if args.method=="R_GAMLP":
        return NARS_R_GAMLP_RDD(in_feats, hidden, num_classes, args.num_hops+1,num_classes,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual)
    elif args.method=="JK_GAMLP":
        return NARS_JK_GAMLP_RDD(in_feats, hidden, num_classes, args.num_hops+1,num_classes,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual)

def gen_model(args,in_size,num_classes):
    if args.method=="R_GAMLP":
        return R_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.act,args.pre_process,args.residual)
    elif args.method=="JK_GAMLP":
        return JK_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.act,args.pre_process,args.residual)

def gen_model_rdd(args,in_size,num_classes):
    if args.method=="R_GAMLP_RDD":
        return R_GAMLP_RDD(in_size, args.hidden, num_classes,args.num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.pre_process,args.residual)

    elif args.method=="JK_GAMLP_RDD":
        return JK_GAMLP_RDD(in_size, args.hidden, num_classes,args.num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.act,args.pre_process,args.residual)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
def train_rdd(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb, predict_prob):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    iter_num=0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att= model(feat_list, label_emb[idx].to(device))
        L1 = loss_fcn(output_att[:len(idx_1)],  y.squeeze(1))
        L2 = F.kl_div(F.log_softmax(output_att[len(idx_1):], dim=1), predict_prob[idx_2].to(device), reduction='batchmean')
        loss = L1 + L2
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    approx_acc = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']
    #print(f'Epoch:{epoch:.4f}, Loss:{loss:.4f}, Train acc:{approx_acc:.4f}')
    return loss, approx_acc

def train(model, feats, labels, loss_fcn, optimizer, train_loader,label_emb):
    model.train()
    device = labels.device
    total_loss = 0
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        output_att=model(batch_feats,label_emb[batch].to(device))
        L1 = loss_fcn(output_att, labels[batch])
        loss_train = L1
        total_loss += loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
@torch.no_grad()
def val(model, feats, labels, val_loader, evaluator,label_emb):
    model.eval()
    device = labels.device
    preds = []
    true=[]
    for batch in val_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats,label_emb[batch].to(device)), dim=-1))
        true.append(labels[batch])
    preds = torch.cat(preds, dim=0)
    true=torch.cat(true,dim=0)
    val_res = evaluator(preds, true)
    return val_res
@torch.no_grad()
def test(model, feats, labels, test_loader, evaluator,
         train_nid, val_nid, test_nid,label_emb):
    model.eval()
    device = labels.device
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats,label_emb[batch].to(device)), dim=-1))
    preds = torch.cat(preds, dim=0)
    train_res = evaluator(preds[:len(train_nid)], labels[:len(train_nid)])
    val_res = evaluator(preds[len(train_nid):len(train_nid)+len(val_nid)], labels[len(train_nid):len(train_nid)+len(val_nid)])
    test_res = evaluator(preds[len(train_nid)+len(val_nid):len(train_nid)+len(val_nid)+len(test_nid)], labels[len(train_nid)+len(val_nid):len(train_nid)+len(val_nid)+len(test_nid)])
    return train_res, val_res, test_res
@torch.no_grad()
def gen_output(model, feats, test_loader,device,label_emb):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(model(batch_feats,label_emb[batch].to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds
