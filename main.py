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
from utils import gen_output,set_seed,train,val,test,gen_model_rdd,gen_model,gen_model_mag_rdd,gen_model_mag
import uuid
import gc
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def run(args, device):
    checkpt_file =f"./output/{args.dataset}/"+uuid.uuid4().hex

    for stage,epochs in enumerate(args.stages):
        if stage>0 and args.rrd:
            predict_prob=torch.load(checkpt_file+f'_{stage}.pt')
            print("This history model Train ACC is {}".format(evaluator.eval({
                    'y_true': labels[:train_node_nums],
                    'y_pred': ensemble_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))

            print("This history model Valid ACC is {}".format(evaluator.eval({
                    'y_true': labels[train_node_nums:train_node_nums+valid_node_nums],
                    'y_pred': ensemble_prob[train_node_nums:train_node_nums+valid_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))

            print("This history model Test ACC is {}".format(evaluator.eval({
                    'y_true': labels[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums],
                    'y_pred': ensemble_prob[train_node_nums+valid_node_nums:train_node_nums+valid_node_nums+test_node_nums].argmax(dim=-1, keepdim=True).cpu()
                })['acc']))
            tr_va_te_nid = torch.arange(total_num_nodes)
            confident_nid = torch.arange(len(predict_prob))[
                    predict_prob.max(1)[0] > args.threshold]
            extra_confident_nid = confident_nid[confident_nid >= len(
                    train_nid)]
            print(f'Stage: {stage}, confident nodes: {len(extra_confident_nid)}')
            enhance_idx = extra_confident_nid
                #enhance_prob = predict_prob[enhance_idx]
            if len(extra_confident_nid) > 0:
                enhance_loader = torch.utils.data.DataLoader(
                        enhance_idx, batch_size=int(args.batch_size*len(enhance_idx)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)
                gc.collect()
        else:
            predict_prob=None
        with torch.no_grad():
            data = prepare_data(device, args,predict_prob)
        feats, labels, in_size, num_classes, \
            train_nid, val_nid, test_nid, evaluator,label_emb = data
        if stage==0:
            train_loader = torch.utils.data.DataLoader(
                 torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=int(args.batch_size*len(train_nid)/(len(enhance_idx)+len(train_nid))), shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid),len(train_nid)+len(val_nid)), batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid)+len(val_nid),len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
            shuffle=False, drop_last=False)
        all_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.batch_size,
            shuffle=False, drop_last=False)
        num_hops = args.num_hops + 1
        if args.use_rdd=='False':
            if args.dataset=="ogbn-products":
                model=gen_model(args,in_size,num_classes)
            elif args.dataset=="ogbn-mag":
                _, num_feats, in_feats = feats[0].shape
                model=gen_model_mag(args,num_feats,in_feats,num_classes)
        else:
            if args.dataset=="ogbn-products":
                model=gen_model_rdd(args,in_size,num_classes)
            elif args.dataset=="ogbn-mag":
                _, num_feats, in_feats = feats[0].shape
                model=gen_model_mag_rdd(args,num_feats,in_feats,num_classes)

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

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            if stage==0:
                train(model, feats, labels, loss_fcn, optimizer, train_loader,label_emb)
            elif stage>0:
                train_rdd(model, train_loader, enhance_loader, optimizer, evaluator, device, feats, labels, label_emb, predict_prob)
            end=time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            if epoch % args.eval_every == 0 and epoch > args.train_num_epochs[stage]:
                with torch.no_grad():
                    acc = val(model, feats, labels, val_loader, evaluator,
                            label_emb)
                end = time.time()
                log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
                log += "Val {:.4f}, ".format(acc)
                if acc > best_val:
                    best_epoch = epoch
                    best_val = acc
                    best_test=val(model, feats, labels, test_loader, evaluator,
                                label_emb)
                    torch.save(model.state_dict(),checkpt_file+f'_{stage}.pkl')
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

        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        preds=gen_output(model,feats,all_loader,labels.device,label_emb)
        torch.save(checkpt_file+f'_{stage}.pt')
    return best_val, best_test,preds


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)
    #set_seed(args.seed)
    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        set_seed(args.seed+i)
        best_val, best_test,preds = run(args, device)
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
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--label-num-hops",type=int,default=9,
                        help="number of hops for label")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed used in the training")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--n-layers-1", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="initial residual parameter for the model")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="initial residual parameter for the model")
    parser.add_argument("--input-drop", type=float, default=0,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--method", type=str, default="JK_GAMLP",
                        help="the model to use")
    parser.add_argument("--use-emb", type=str)
    parser.add_argument("--use-relation-subsets", type=str)
    parser.add_argument("--use-rdd", action='store_true', default=False,
                        help="whether to use the reliable data distillation")
    parser.add_argument("--train-num-epochs", nargs='+',type=int, default=[100, 100],
                        help="The Train epoch setting for each stage.")
    parser.add_argument("--stages", nargs='+',type=int, default=[300, 300],
                        help="The epoch setting for each stage.")
    args = parser.parse_args()
    print(args)
    main(args)
