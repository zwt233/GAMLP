import random

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

eps = 1e-9

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def entropy(probs):
    res = - probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
    return res

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def to_scipy(tensor):
    """Convert a sparse tensor to scipy matrix"""
    values = tensor._values()
    indices = tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def from_scipy(sparse_mx):
    """Convert a scipy sparse matrix to sparse tensor"""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_spectral_emb(adj, K):
    A = to_scipy(adj.to("cpu"))
    L = from_scipy(sp.csgraph.laplacian(A, normed=True))
    _, spectral_emb = torch.lobpcg(L, K)
    return spectral_emb.to(adj.device)

def generate_subset_list(g, num_subsets, target_ntype="paper"):
    edges = {e:(u,v) for u,v,e in g.metagraph().edges}
    print(edges)
    all_relations = list(edges.keys())
    subset_list = []
    while len(subset_list) < num_subsets:
        touched = False
        candidate = []
        for relation in all_relations:
            p = np.random.rand()
            if p >= 0.5:
                candidate.append(relation)
                if target_ntype in edges[relation]:
                    touched = True
        if touched:
            candidate = tuple(candidate)
            if candidate not in subset_list:
                subset_list.append(candidate)
    return subset_list


# Following part adapted from NARS: https://github.com/facebookresearch/NARS/blob/main/utils.py

###############################################################################
# Evaluator for different datasets
###############################################################################

def batched_acc(pred, labels):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)


def get_evaluator(dataset):
    dataset = dataset.lower()
    if dataset.startswith("oag"):
        return batched_ndcg_mrr
    else:
        return batched_acc


def compute_mean(metrics, nid):
    num_nodes = len(nid)
    return [m[nid].float().sum().item() / num_nodes for  m in metrics]


###############################################################################
# Original implementation of evaluation metrics NDCG and MRR by HGT author
# https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/pyHGT/utils.py
###############################################################################

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def ndcg_mrr(pred, labels):
    """
    Compute both NDCG and MRR for single-label and multi-label. Code extracted from
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_venue.py#L316 (single label)
    and
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_field.py#L322 (multi-label)
    """
    test_res = []
    if len(labels.shape) == 1:
        # single-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [(bi == ai).int().tolist()]
    else:
        # multi-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [ai[bi].int().tolist()]
    ndcg = np.mean([ndcg_at_k(resi, len(resi)) for resi in test_res])
    mrr = mean_reciprocal_rank(test_res)
    return ndcg, mrr


###############################################################################
# Fast re-implementation of NDCG and MRR for a batch of nodes.
# We provide unit test below using random input to verify correctness /
# equivalence.
###############################################################################

def batched_dcg_at_k(r, k):
    assert(len(r.shape) == 2 and r.size != 0 and k > 0)
    r = r[:, :k].float()
    # Usually, one defines DCG = \sum\limits_{i=0}^{n-1}\frac{r_i}/{log2(i+2)}
    # So, we should
    # return (r / torch.log2(torch.arange(0, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 2)).sum(dim=1)
    # However, HGT author implements DCG = r_0 + \sum\limits_{i=1}^{n-1}\frac{r_i}/{log2(i+1)}, which makes DCG and NDCG larger
    # Here, we follow HGT author for a fair comparison
    return r[:, 0] + (r[:, 1:] / torch.log2(torch.arange(1, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 1)).sum(dim=1)


def batched_ndcg_at_k(r, k):
    dcg_max = batched_dcg_at_k(r.sort(dim=1, descending=True)[0], k)
    dcg_max_inv = 1.0 / dcg_max
    dcg_max_inv[torch.isinf(dcg_max_inv)] = 0
    return batched_dcg_at_k(r, k) * dcg_max_inv


def batched_mrr(r):
    r = r != 0
    # torch 1.5 does not guarantee max returns first occurrence
    # https://pytorch.org/docs/1.5.0/torch.html?highlight=max#torch.max
    # So we get first occurrence of non-zero using numpy max
    max_indices = torch.from_numpy(r.cpu().numpy().argmax(axis=1))
    max_values = r[torch.arange(r.shape[0]), max_indices]
    r = 1.0 / (max_indices.float() + 1)
    r[max_values == 0] = 0
    return r


def batched_ndcg_mrr(pred, labels):
    pred = pred.argsort(descending=True)
    if len(labels.shape) == 1:
        # single-label
        labels = labels.view(-1, 1)
        rel = (pred == labels).int()
    else:
        # multi-label
        rel = torch.gather(labels, 1, pred)
    return batched_ndcg_at_k(rel, rel.shape[1]), batched_mrr(rel)

