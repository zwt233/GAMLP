import math
import os
import random
import time
import gc

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from dataset import load_dataset
from utils import entropy

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    if args.aggr_gpu < 0:
        device = "cpu"
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.K + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.K}").cpu())
    del new_g, feat_dict, new_edges
    gc.collect()
    torch.cuda.empty_cache()
    return res


def neighbor_average_features(g, feat, args, use_norm=False, style="all"):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats", style)
    
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    g = g.to(aggr_device)
    
    feat = feat.to(aggr_device)

    if style == "all":
        g.ndata['feat_0'] = feat
            
        # print(g.ndata["feat"].shape)
        # print(norm.shape)
        if use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.K + 1):
            g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop-1}']
            # g.ndata['pre_label_emb'] = g.ndata['label_emb']
            if use_norm:
                g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop}'] * norm
                
                g.update_all(fn.copy_src(src=f'feat_{hop}', out='msg'),
                            fn.sum(msg='msg', out=f'feat_{hop}'))
                g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop}'] * norm
            else:
                g.update_all(fn.copy_src(src=f'feat_{hop}', out='msg'),
                            fn.mean(msg='msg', out=f'feat_{hop}'))

                
            # if hop > 1:
            #     g.ndata['label_emb'] = 0.5 * g.ndata['pre_label_emb'] + \
            #                            0.5 * g.ndata['label_emb']
        res = []
        for hop in range(args.K + 1):
            res.append(g.ndata.pop(f'feat_{hop}'))
        gc.collect()

        # if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                            dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res

    # del g.ndata['pre_label_emb']
    elif style in ["last", "ppnp"]:
        
        if style == "ppnp": init_feat = feat
        if use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.label_K+1):         
            # g.ndata["f_next"] = g.ndata["f"]
            if use_norm:
                feat = feat * norm
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.sum(msg='msg', out='f'))
                feat = g.ndata.pop('f')
                # degs = g.in_degrees().float().clamp(min=1)
                # norm = torch.pow(degs, -0.5)
                # shp = norm.shape + (1,) * (g.ndata['f'].dim() - 1)
                # norm = torch.reshape(norm, shp)
                feat = feat * norm
            else:
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
                feat = g.ndata.pop('f')
            if style == "ppnp":
                feat = 0.5 * feat + 0.5 * init_feat
            
        res = feat
        gc.collect()

        # if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = torch.zeros((num_target,) + feat.shape[1:],
                                dtype=feat.dtype, device=feat.device)
        new_res[target_ids] = res[target_mask]
        res = new_res

    return res

def prepare_data(device, args, probs_path, subset_list, stage=0):
    """
    Load dataset and compute neighbor-averaged node features used by scalable GNN model
    Note that we select only one integrated representation as node feature input for mlp 
    """
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))

    data = load_dataset(aggr_device, args)
    t1 = time.time()
    
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    target_type_id = g.get_ntype_id("paper")
    homo_g = dgl.to_homogeneous(g, ndata=["feat"])
    homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
    # homo_g = dgl.add_self_loop(homo_g)
    homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
    feat_averaging_style = "all" if args.model in ["sagn", "plain_sagn", "simple_sagn", "sign"] else "ppnp"
    label_averaging_style = "last"
    in_feats = g.ndata['feat']['paper'].shape[1]
    # n_classes = (labels.max() + 1).item() if labels.dim() == 1 else labels.size(1)
    print("in_feats:", in_feats)
    feat = g.ndata['feat']['paper']

    if stage > 0:
        teacher_probs = torch.load(probs_path, map_location=aggr_device)
        tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)

        # assert len(teacher_probs) == len(feat)
        if args.dataset in ['oag_L1']:
            threshold = - args.threshold * np.log(args.threshold) - (1-args.threshold) * np.log(1-args.threshold)
            entropy_distribution = entropy(teacher_probs)
            print(threshold)
            print(entropy_distribution.mean(1).max().item())
            
            confident_nid_inner = torch.arange(len(teacher_probs))[(entropy_distribution.mean(1) <= threshold)]
        else:
            confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > args.threshold]
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(train_nid)]
        confident_nid = tr_va_te_nid[confident_nid_inner]
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]
        print(f"pseudo label number: {len(confident_nid)}")
        if args.dataset in ['oag_L1']:
            pseudo_labels = teacher_probs
            pseudo_labels[pseudo_labels >= 0.5] = 1
            pseudo_labels[pseudo_labels < 0.5] = 0
            labels_with_pseudos = torch.ones_like(labels)
        else:
            pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
            labels_with_pseudos = torch.zeros_like(labels)
        train_nid_with_pseudos = np.union1d(train_nid, confident_nid)
        print(f"enhanced train set number: {len(train_nid_with_pseudos)}")
        labels_with_pseudos[train_nid] = labels[train_nid]
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid_inner]
        
        # train_nid_with_pseudos = np.random.choice(train_nid_with_pseudos, size=int(0.5 * len(train_nid_with_pseudos)), replace=False)
    else:
        teacher_probs = None
        pseudo_labels = None
        labels_with_pseudos = labels.clone()
        confident_nid = train_nid
        train_nid_with_pseudos = train_nid
    
    if args.use_labels:
        print("using label information")
        if args.dataset in ['oag_L1']:
            label_emb = 0.5 * torch.ones([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = labels_with_pseudos.mean(0).repeat([feat.shape[0], 1])
            label_emb[train_nid_with_pseudos] = labels_with_pseudos.float()[train_nid_with_pseudos]
        else:
            label_emb = torch.zeros([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = (1. / n_classes) * torch.ones([feat.shape[0], n_classes]).to(device)
            label_emb[train_nid_with_pseudos] = F.one_hot(labels_with_pseudos[train_nid_with_pseudos], num_classes=n_classes).float().to(labels.device)


        # if args.dataset == "ogbn-mag":
            # rand_weight = torch.Tensor(n_classes, 128).uniform_(-0.5, 0.5)
            # label_emb = torch.matmul(label_emb, rand_weight.to(device))
            # pca = PCA(n_components=128)
            # label_emb = torch.FloatTensor(pca.fit_transform(label_emb.cpu())).to(device)
        target_mask = homo_g.ndata["target_mask"]
        target_ids = homo_g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_label_emb = torch.zeros((len(homo_g.ndata["feat"]),) + label_emb.shape[1:],
                            dtype=label_emb.dtype, device=label_emb.device)
        new_label_emb[target_mask] = label_emb[target_ids]
        label_emb = new_label_emb
    else:
        label_emb = None

    if label_emb is not None:
        label_emb = neighbor_average_features(homo_g, label_emb, args, use_norm=False, style=label_averaging_style)

    del homo_g
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    feats = {("raw",): [feat]}
    for rel_subset in subset_list:
        feats[rel_subset] = gen_rel_subset_feature(g, rel_subset, args, device)

    # if args.save_temporal_emb:
    #     torch.save(feats, emb_path)
        
    
    # save smoothed node features and initial smoothed node label embeddings, 
    # if "load" is set true and they have not been saved
 
    # label_emb = (label_emb - label_emb.mean(0)) / label_emb.std(0)
    # eval_feats = neighbor_average_features(g, eval_feat, args)
    labels = labels.to(device)
    labels_with_pseudos = labels_with_pseudos.to(device)
    # move to device

    train_nid = train_nid.to(device)
    train_nid_with_pseudos = torch.LongTensor(train_nid_with_pseudos).to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    t2 = time.time()

    return feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, evaluator, t2 - t1
