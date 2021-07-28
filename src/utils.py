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
def get_model(args):
    return 0
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def train(model, feats, labels, loss_fcn, optimizer, train_loader,epoch,args):
    model.train()
    device = labels.device
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        output_att=model(batch_feats)
        L1 = loss_fcn(output_att, labels[batch])
#        L2 = loss_kd_only(output_att,teacher_output[batch],args.temp)
        loss_train = L1
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()   
@torch.no_grad()
def val(model, feats, labels, val_loader, evaluator,
         val_nid):
    model.eval()
    device = labels.device
    preds = []
    for batch in val_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats), dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    val_res = evaluator(preds, labels[val_nid])
    return val_res
@torch.no_grad()        
def test(model, feats, labels, test_loader, evaluator,
         train_nid, val_nid, test_nid):
    model.eval()
    device = labels.device
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats), dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    return train_res, val_res, test_res
    
def gmlp_train(model, feats, labels, loss_fcn, optimizer, train_loader,epoch,epochs):
    model.train()
    device = labels.device

    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        output_att, output_concat=model(batch_feats)
        L1 = loss_fcn(output_att, labels[batch])
        L2 = loss_fcn(output_concat, labels[batch])
        loss_train = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2        
        #loss = loss_fcn(model(batch_feats), labels[batch])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()    
@torch.no_grad()
def gmlp_val(model, feats, labels, val_loader, evaluator,
         val_nid):
    model.eval()
    device = labels.device
    preds = []
    for batch in val_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats)[0], dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    val_res = evaluator(preds, labels[val_nid])
    return val_res
@torch.no_grad()        
def gmlp_test(model, feats, labels, test_loader, evaluator,
         train_nid, val_nid, test_nid):
    model.eval()
    device = labels.device
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats)[0], dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    return train_res, val_res, test_res

@torch.no_grad()
def gen_output(model, feats, test_loader,device):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(model(batch_feats).cpu().numpy())
    # Concat mini-batch prediction results along node dimension
    preds = np.concatenate(preds, axis=0)
    return preds