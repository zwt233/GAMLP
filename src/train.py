import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from load_dataset import prepare_data
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
from utils import gen_output,set_seed,train,val,test,gmlp_train,gmlp_val,gmlp_test
from model import GMLP,R_GATE

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def run(args, data, device):
    checkpt_file =f"output/{args.dataset}/"+"gmlp"+'.pt'
    feats, labels, in_size, num_classes, \
        train_nid, val_nid, test_nid, evaluator = data
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_nid, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)    
    test_loader = torch.utils.data.DataLoader(
        torch.arange(labels.shape[0]), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    num_hops = args.R + 1
    num_node=feats[0].shape[0]  
    
    model= R_GATE(in_size, args.num_hidden, num_classes, num_hops,
                 args.dropout, args.input_dropout,args.att_dropout,args.alpha,
                 args.part_1_layers,args.part_2_layers,num_node,args.pre_process)
    print(model)                                                      
    model = model.to(device)
    print("# Params:", get_n_params(model))

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    record={}    
    count=0
    
    for epoch in range(args.num_epochs):
        start = time.time()
        gmlp_train(model, feats, labels, loss_fcn, optimizer, train_loader,epoch,args.num_epochs)
        end=time.time()
        log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
        if epoch % args.eval_every == 0 and epoch > args.train_num_epochs:
            with torch.no_grad():
                if args.double:
                  acc=gmlp_val(model, feats, labels, val_loader, evaluator,
                           val_nid)
                else:
                  acc = val(model, feats, labels, val_loader, evaluator,
                           val_nid)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Val {:.4f}, ".format(acc)
            if acc > best_val:
                best_epoch = epoch
                best_val = acc
                if args.double:
                   accs=gmlp_test(model, feats, labels, test_loader, evaluator,
                             train_nid, val_nid, test_nid)    
                else:
                   accs=test(model, feats, labels, test_loader, evaluator,
                             train_nid, val_nid, test_nid)                     
                best_test=accs[2]                   
                torch.save(model.state_dict(),checkpt_file)
                count=0
            else:
                count=count+args.eval_every
                if count>=args.patience:
                    break
            log+="Best Epoch {},Val {:.4f}, Test {:.4f}".format(
                        best_epoch,best_val,best_test)
        print(log)        

    
    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))

    model.load_state_dict(torch.load(checkpt_file))
    preds=gen_output(model,feats,test_loader,labels.device)
    return best_val, best_test,preds


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)
    #set_seed(args.seed)
    with torch.no_grad():
        data = prepare_data(device, args)
    
    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        set_seed(i)
        best_val, best_test,preds = run(args, data, device)
        np.save(f"output/{args.dataset}/output_{i}.npy",preds)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
          f"std: {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
          f"std: {np.std(test_accs):.4f}")
    return np.mean(test_accs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMLP")
    parser.add_argument("--num-epochs", type=int, default=1500)
    parser.add_argument("--train-num-epochs", type=int, default=1000)    
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--R", type=int, default=15,
                        help="number of hops")
    parser.add_argument("--seed", type=int, default=0,
                        help="number of hops")                        
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=10000,
                        help="evaluation batch size")
    parser.add_argument("--part-1-layers", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--part-2-layers", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=400,
                        help="early stop of times of the experiment")       
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="early stop of times of the experiment")             
    parser.add_argument("--input-dropout", type=float, default=0,
                        help="early stop of times of the experiment")  
    parser.add_argument("--att-dropout", type=float, default=0.5,
                        help="early stop of times of the experiment")                          
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="early stop of times of the experiment")                                  
    parser.add_argument("--double", action='store_true', default=False,
                        help="indicate the model is gmlp")  
                                                 
    args = parser.parse_args()
    print(args)
    main(args)
