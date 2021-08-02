import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from heteo_data import load_data, read_relation_subsets, gen_rel_subset_feature,preprocess_features
import torch.nn.functional as F
import gc

def prepare_label_emb(args,label_teacher_emb=None):
     from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
     dataset = DglNodePropPredDataset(name=args.dataset, root='/data4/zwt/')
     splitted_idx = dataset.get_idx_split()
     train_idx = splitted_idx["train"]
     valid_idx = splitted_idx["valid"]
     test_idx = splitted_idx["test"]
     g, labels = dataset[0]
     n_classes = dataset.num_classes
     if args.dataset=='ogbn-100Mpapers':
         g = dgl.add_reverse_edges(g, copy_ndata=True)
     elif args.dataset=='ogbn-mag':
        target_type_id = g.get_ntype_id("paper")
        homo_g = dgl.to_homogeneous(g, ndata=["feat"])
        homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
        homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
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
     if args.dataset=='ogbn-mag':
         target_mask = homo_g.ndata["target_mask"]
         target_ids = homo_g.ndata[dgl.NID][target_mask]
         num_target = target_mask.sum().item()
         new_label_emb = torch.zeros((len(homo_g.ndata["feat"]),) + label_emb.shape[1:],
                            dtype=label_emb.dtype, device=label_emb.device)
         new_label_emb[target_mask] = label_emb[target_ids]
         label_emb = new_label_emb
         g=homo_g
     for hop in range(args.label_num_hops):
         y = neighbor_average_labels(g, y, args)
         gc.collect()
     res=y
     if args.dataset=='ogbn-mag':
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = torch.zeros((num_target,) + feat.shape[1:],
                                dtype=feat.dtype, device=feat.device)
        new_res[target_ids] = res[target_mask]
        res = new_res
     return torch.cat([res[train_idx],res[valid_idx],res[test_idx]],dim=0)
def neighbor_average_labels(g,feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels")
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                     fn.mean("msg", "f"))
    feat=g.ndata.pop('f')
    if args.dataset == "ogbn-mag":
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
    return feat

def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res
def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]
def load_dataset(name, device):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name,root="/data4/zwt/")
    splitted_idx = dataset.get_idx_split()
    if name == "ogbn-products":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        g.ndata["labels"] = labels
        g.ndata['feat'] = g.ndata['feat'].float()
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-mag":
        data = load_data(device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid = data
        evaluator = get_ogb_evaluator(name)


    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")
    return g, torch.cat([labels[train_nid],labels[val_nid],labels[test_nid]]), n_classes, train_nid, val_nid, test_nid, evaluator

def prepare_data(device, args,teacher_probs=None):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    label_emb=None
    if args.use_rdd:
        label_emb=prepare_label_emb(args,teacher_probs)
    data = load_dataset(args.dataset, device)

    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    if args.dataset=='ogbn-products':
        feats=neighbor_average_features(g,args)
        in_feats=feats[0].shape[1]
    elif args.dataset=='ogbn-mag':
        rel_subsets = read_relation_subsets(args.use_relation_subsets)
        with torch.no_grad():
            feats = preprocess_features(g, rel_subsets, args, device)
            print("Done preprocessing")
        _, num_feats, in_feats = feats[0].shape
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    return feats, labels, in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator,label_emb
