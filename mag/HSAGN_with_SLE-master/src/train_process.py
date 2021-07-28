import os
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_mean

def gmlp_train(model,feats,label_emb,teacher_probs,labels,loss_fcn,optimizer,train_loader,epoch,epochs):
    model.train()
    device = labels.device
    for batch in train_loader:
        if len(batch) == 1:
            continue
        batch_feats= {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb= label_emb[batch].to(device)
        else:
            batch_label_emb=None
        out1,out2=model(batch_feats,batch_label_emb)
        L1 = loss_fcn(out1,  labels[batch])
        L2 = loss_fcn(out2,  labels[batch])
        loss = L1 + np.cos(np.pi * epoch / (2 * epochs)) * L2
        loss.backward()
        optimizer.step()
def train(model, feats, label_emb, teacher_probs, labels, loss_fcn, optimizer, train_loader):
    model.train()
    device = labels.device
    for batch in train_loader:

        if len(batch) == 1:
            continue
        batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None

        out, _ = model(batch_feats, batch_label_emb)

        loss = loss_fcn(out, labels[batch])
                # + 1 * (torch.norm(model.clf.hop_attn_l, p=1) + torch.norm(model.clf.hop_attn_r, p=1))
        # We also tried KD loss, but it brings no improvements empirically.
        # T = 0.5
        # alpha = 0.5
        # if teacher_probs is not None:
        #     loss = (1-alpha) * loss + alpha * F.kl_div(F.log_softmax(out, dim=-1) / T, teacher_probs[batch] / T) * (T * T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def gmlp_test(model,feats,label_emb,teachers_probs,labels,loss_fcn,val_loader,test_loader,evaluator,train_nid,val_nid,test_nid):
    model.eval()
    num_nodes = labels.shape[0]
    device = labels.device
    loss_list = []
    count_list = []
    scores = []
    for batch in val_loader:
        batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        # We can get attention scores from SAGN
        out, _ = model(batch_feats, batch_label_emb)
        loss_list.append(loss_fcn(out, labels[batch]).cpu().item())
        count_list.append(len(batch))
    loss_list = np.array(loss_list)
    count_list = np.array(count_list)
    val_loss = (loss_list * count_list).sum() / count_list.sum()
    start = time.time()
    for batch in test_loader:
        batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        out, _ = model(batch_feats, batch_label_emb)

        scores.append(evaluator(out, labels[batch]))

    # Concat mini-batch prediction results along node dimension
    metrics = [torch.cat(s, dim=0) for s in zip(*scores)]
    end = time.time()
    train_res = compute_mean(metrics, np.arange(len(train_nid)))
    val_res = compute_mean(metrics, len(train_nid) + np.arange(len(val_nid)))
    test_res = compute_mean(metrics, len(train_nid) + len(val_nid) + np.arange(len(test_nid)))
    # train_res = evaluator(preds[:len(train_nid)], labels[train_nid])
    # val_res = evaluator(preds[len(train_nid):(len(train_nid)+len(val_nid))], labels[val_nid])
    # test_res = evaluator(preds[(len(train_nid)+len(val_nid)):], labels[test_nid])
    return train_res, val_res, test_res, val_loss, end - start
def test(model, feats, label_emb, teacher_probs, labels, loss_fcn, val_loader, test_loader, evaluator,
         train_nid, val_nid, test_nid):
    model.eval()
    num_nodes = labels.shape[0]
    device = labels.device
    loss_list = []
    count_list = []
    scores = []
    for batch in val_loader:
        batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        # We can get attention scores from SAGN
        out, _ = model(batch_feats, batch_label_emb)
        loss_list.append(loss_fcn(out, labels[batch]).cpu().item())
        count_list.append(len(batch))
    loss_list = np.array(loss_list)
    count_list = np.array(count_list)
    val_loss = (loss_list * count_list).sum() / count_list.sum()
    start = time.time()
    for batch in test_loader:
        batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        out, _ = model(batch_feats, batch_label_emb)

        scores.append(evaluator(out, labels[batch]))

    # Concat mini-batch prediction results along node dimension
    metrics = [torch.cat(s, dim=0) for s in zip(*scores)]
    end = time.time()
    train_res = compute_mean(metrics, np.arange(len(train_nid)))
    val_res = compute_mean(metrics, len(train_nid) + np.arange(len(val_nid)))
    test_res = compute_mean(metrics, len(train_nid) + len(val_nid) + np.arange(len(test_nid)))
    # train_res = evaluator(preds[:len(train_nid)], labels[train_nid])
    # val_res = evaluator(preds[len(train_nid):(len(train_nid)+len(val_nid))], labels[val_nid])
    # test_res = evaluator(preds[(len(train_nid)+len(val_nid)):], labels[test_nid])
    return train_res, val_res, test_res, val_loss, end - start
