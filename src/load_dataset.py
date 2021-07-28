import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator  
def D_neighbor_average_n2v(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["n2v_0"] = g.ndata["n2v"]
    for hop in range(1, args.R + 1):
        g.ndata['h']=g.ndata[f"n2v_{hop-1}"]*g.ndata['norm']*g.ndata['norm'] 
        g.update_all(fn.copy_u("h", "msg"),
                     fn.sum("msg", "h"))
        h=g.ndata.pop('h')
        g.ndata[f"n2v_{hop}"]=h 
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"n2v_{hop}"))
    return res
def sys_neighbor_average_n2v(g,args):
    print("Compute neighbor-averaged feats")
    g.ndata["n2v_0"] = g.ndata["n2v"]
    for hop in range(1, args.R + 1):        
        g.ndata['h']=g.ndata[f"n2v_{hop-1}"]*g.ndata['norm']
        g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h=g.ndata.pop('h')*g.ndata['norm']
        g.ndata[f"n2v_{hop}"]=h 
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"n2v_{hop}"))
    return res
def sys_neighbor_average_features(g,args):
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):        
        g.ndata['h']=g.ndata[f"feat_{hop-1}"]*g.ndata['norm']
        g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h=g.ndata.pop('h')*g.ndata['norm']
        g.ndata[f"feat_{hop}"]=h 
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res
def D_neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.ndata['h']=g.ndata[f"feat_{hop-1}"]*g.ndata['norm']*g.ndata['norm'] 
        g.update_all(fn.copy_u("h", "msg"),
                     fn.sum("msg", "h"))
        h=g.ndata.pop('h')
        g.ndata[f"feat_{hop}"]=h 
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))

    return res
def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.R + 1):
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
    if name not in ["ogbn-products", "ogbn-arxiv"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
      
    dataset = DglNodePropPredDataset(name=name,root='/home2/zwt/')
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    g.ndata["labels"] = labels
    if name == "ogbn-arxiv":
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = g.ndata['feat'].float() 
    else:
        g.ndata['feat'] = g.ndata['feat'].float()
    n_classes = dataset.num_classes
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)

    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")
    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(args.dataset, device)
    
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    #print(type(g)) DGL graph
    #print(type(labels)) tensor
    #in_feats = g.ndata['feat'].shape[1]
    degs = g.in_degrees().float()    
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)    
    if args.dataset=='ogbn-arxiv':
        feats_sys=sys_neighbor_average_features(g, args)
        feats_d=D_neighbor_average_features(g, args)
        '''
        if args.n2v:
            g.ndata["n2v"]=torch.load("n2v_embedding.pt")
            n2v_sys=sys_neighbor_average_n2v(g,args)
            n2v_d=D_neighbor_average_n2v(g,args)
            feats=[]
            for i in range(len(feats_sys)):
                feats.append(torch.cat([feats_sys[i],feats_d[i],n2v_sys[i],n2v_d[i]],dim=1))
        else:
        '''
        feats=[]
        for i in range(len(feats_sys)):
            feats.append(torch.cat([feats_sys[i],feats_d[i]],dim=1))        
    else:
        feats=neighbor_average_features(g,args)
    in_feats=feats[0].shape[1]
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    return feats, labels, in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator