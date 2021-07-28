import argparse
import gc
import math
import os
import random
import time
from copy import deepcopy

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset, read_relation_subsets
from gen_models import get_model
from preprocessing import prepare_data
from train_process import test, train,gmlp_train
from utils import generate_subset_list, get_n_params, seed

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# sample_subset_list = [("writes", "cites"),
#                ("affiliated_with", "has_topic"),
#                ("affiliated_with", "writes"),
#                ("affiliated_with", "writes", "cites", "has_topic"),
#                ("writes",),
#                ("affiliated_with", "cites"),
#                ("cites", "has_topic"),
#                ("affiliated_with", "writes", "cites")]
# all_relations = ["cites", "writes", "affiliated_with", "has_topic"]

def multilabel_loss(input, labels):
    return F.kl_div(F.log_softmax(input, dim=-1), labels, reduction="batchmean")

def run(args, data, subset_list, device, stage=0):
    feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, evaluator, _ = data
    train_loader = torch.utils.data.DataLoader(
        train_nid.cpu(), batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2)
    train_loader_with_pseudos = torch.utils.data.DataLoader(
        train_nid_with_pseudos.cpu(), batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_nid.cpu(), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        torch.cat([train_nid, val_nid, test_nid], dim=0).cpu(), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False, num_workers=2)
    all_loader = torch.utils.data.DataLoader(
        torch.arange(len(labels)), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    # Initialize model and optimizer for each run

    model = get_model(in_feats, n_classes, stage, subset_list, args)
    model = model.to(device)
    print("# Params:", get_n_params(model))

    if len(labels.shape) == 1:
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = multilabel_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    best_val_ = 0
    best_val_loss = 1e9
    best_test = 0
    num_epochs = args.epoch_setting[stage]
    train_time = []
    inference_time = []
    val_accs = []
    val_loss = []


    for epoch in range(1, num_epochs + 1):
        start = time.time()
        if args.model=="NARS_gmlp":
            gmlp_train(model, feats, label_emb, teacher_probs, labels_with_pseudos, loss_fcn, optimizer, train_loader_with_pseudos,epoch,args.epochs )
        else:
            train(model, feats, label_emb, teacher_probs, labels_with_pseudos, loss_fcn, optimizer, train_loader_with_pseudos )
        med = time.time()

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, label_emb, teacher_probs, labels, loss_fcn, val_loader, test_loader, evaluator,
                           train_nid, val_nid, test_nid)
            end = time.time()

            if (acc[1][0] > best_val_ and args.acc_loss == "acc") or (acc[3] < best_val_loss and args.acc_loss == "loss"):
                best_epoch = epoch
                best_val = acc[1]
                best_val_ = acc[1][0]
                best_test = acc[2]
                best_val_loss = acc[3]
                best_model = deepcopy(model)

            train_time.append(med - start)
            inference_time.append(acc[-1])
            val_accs.append(acc[1][0])
            val_loss.append(acc[-2])
            log = "Epoch {}, Time(s): {:.4f} {:.4f}, ".format(epoch, med - start, acc[-1])
            if args.dataset.startswith("oag"):
                log += "Best Val loss: {:.4f}\n".format(best_val_loss)
                log += "NDCG: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}\n".format(acc[0][0], acc[1][0], acc[2][0], best_val[0], best_test[0])
                log += " MRR: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(acc[0][1], acc[1][1], acc[2][1], best_val[1], best_test[1])
            else:
                log += "Best Val loss: {:.4f}, Accs: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(best_val_loss, acc[0][0], acc[1][0], acc[2][0], best_val[0], best_test[0])
            print(log)

    if args.dataset.startswith("oag"):
        print("Stage: {}, Best Epoch {}, Val [{:.4f}, {:.4}], Test [{:.4f}, {:.4f}]".format(
            stage, best_epoch, best_val[0], best_val[1], best_test[0], best_test[1]))
    else:
        print("Stage: {}, Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            stage, best_epoch, best_val[0], best_test[0]))
    with torch.no_grad():
        best_model.eval()
        probs = []
        if "sagn" in args.model and (not args.avoid_features):
            attn_weights = []
        else:
            attn_weights = None
        for batch in test_loader:
            batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
            if label_emb is not None:
                batch_label_emb = label_emb[batch].to(device)
            else:
                batch_label_emb = None
            out, a = best_model(batch_feats, batch_label_emb)
            if args.dataset in ['oag_L1']:
                out = out.sigmoid()
            else:
                out = out.softmax(dim=1)
            # remember to transfer output probabilities to cpu
            probs.append(out.cpu())
            if "sagn" in args.model and (not args.avoid_features):
                attn_weights.append(a.cpu().squeeze(1).squeeze(1))
        probs = torch.cat(probs, dim=0)
        if "sagn" in args.model and (not args.avoid_features):
            attn_weights = torch.cat(attn_weights)

    del model, best_model
    del feats, label_emb, teacher_probs, labels, labels_with_pseudos
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return best_val, best_test, probs, train_time, inference_time, val_accs, val_loss, attn_weights


def main(args):
    device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else f"cuda:{args.aggr_gpu}")
    # torch.cuda.set_device(device)
    # initial_emb_path = os.path.join("..", "embeddings", args.dataset,
                            # args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                            # "initial_smoothed_features.pt")

    total_best_val_accs = []
    total_best_test_accs = []
    total_val_accs = []
    total_val_losses = []
    total_preprocessing_times = []
    total_train_times = []
    total_inference_times = []

    for i in range(args.num_runs):
        print("-" * 100)
        print(f"Run {i} start training")
        seed(seed=args.seed + i)
        # pre-load dataset metagraph to generate subsets
        if args.fixed_subsets:
            subset_list = read_relation_subsets(args.dataset, args.example_subsets_path)
        else:
            g, _, _, _, _, _, _ = load_dataset(aggr_device, args)
            subset_list = generate_subset_list(g, args.sample_size)


        print(f"subset list:\n {subset_list}")

        best_val_accs = []
        best_test_accs = []
        val_accs = []
        val_losses = []

        preprocessing_times = []
        train_times = []
        inference_times = []

        for stage in range(len(args.epoch_setting)):

            if args.warmup_stage > -1:
                if stage <= args.warmup_stage:
                    probs_path = os.path.join(args.probs_dir, args.dataset, args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                                    f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_probs_seed_{args.seed + i}_stage_{stage}.pt')
                    if os.path.exists(probs_path):
                        print(f"bypass stage {stage} since warmup_stage is setted and associated file exists.")
                        continue
            print("-" * 100)
            print(f"Stage {stage} start training")
            if stage > 0:
                probs_path = os.path.join(args.probs_dir, args.dataset,
                                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_probs_seed_{args.seed + i}_stage_{stage - 1}.pt')
            else:
                probs_path = ''
            with torch.no_grad():
                data = prepare_data(device, args, probs_path, subset_list, stage=stage)
            preprocessing_times.append(data[-1])
            print(f"Preprocessing costs {(data[-1]):.4f} s")
            best_val, best_test, probs, train_time, inference_time, val_acc, val_loss, attn_weights = run(args, data, subset_list, device, stage)
            train_times.append(train_time)
            inference_times.append(inference_time)
            if stage > 0 and args.remove_intermediate_probs:
                os.remove(probs_path)

            val_accs.append(val_acc)
            val_losses.append(val_loss)
            new_probs_path = os.path.join(args.probs_dir, args.dataset,
                                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_probs_seed_{args.seed + i}_stage_{stage}.pt')
            if not os.path.exists(os.path.dirname(new_probs_path)):
                os.makedirs(os.path.dirname(new_probs_path))
            torch.save(probs, new_probs_path)
            best_val_accs.append(best_val)
            best_test_accs.append(best_test)

            path = os.path.join("../converge_stats", args.dataset,
                                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                                f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.csv")
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            # print(val_acc)
            '''
            df = pd.DataFrame()
            df['epoch'] = np.arange(args.eval_every, args.epoch_setting[stage] + 1, args.eval_every)
            df['val_acc'] = val_acc
            df.to_csv(path)
            fig_path = os.path.join("../converge_stats", args.dataset,
                        args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                        f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.png")
            sns.set()
            line_plt = sns.lineplot(data=df, x='epoch', y='val_acc')
            line = line_plt.get_figure()
            line.savefig(fig_path)
            plt.close()

            path = os.path.join("../converge_stats", args.dataset,
                                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                                f"val_loss_use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.csv")
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            # print(val_loss)
            df = pd.DataFrame()
            df['epoch'] = np.arange(args.eval_every, args.epoch_setting[stage] + 1, args.eval_every)
            df['val_loss'] = val_loss
            df.to_csv(path)
            fig_path = os.path.join("../converge_stats", args.dataset,
                        args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                        f"val_loss_use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.png")
            sns.set()
            line_plt = sns.lineplot(data=df, x='epoch', y='val_loss')
            line = line_plt.get_figure()
            line.savefig(fig_path)
            plt.close()

            if ("sagn" in args.model)  and (not args.avoid_features):
                path = os.path.join("../attn_weights", args.dataset, args.model,
                    f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.csv")
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                df = pd.DataFrame(data=attn_weights.cpu().numpy())
                df.to_csv(path)
                fig_path = os.path.join("../attn_weights", args.dataset, args.model,
                    f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_seed_{args.seed + i}_stage_{stage}.png")
                sns.set()
                heatmap_plt = sns.heatmap(df)
                heatmap = heatmap_plt.get_figure()
                heatmap.savefig(fig_path)
                plt.close()

            '''
            del data, probs, attn_weights

            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        total_best_val_accs.append(best_val_accs)
        total_best_test_accs.append(best_test_accs)
        total_val_accs.append(val_accs)
        total_val_accs.append(val_losses)
        total_preprocessing_times.append(preprocessing_times)
        total_train_times.append(train_times)
        total_inference_times.append(inference_times)

    total_best_val_accs = np.array(total_best_val_accs)
    total_best_test_accs = np.array(total_best_test_accs)
    total_val_accs = np.array(total_val_accs)
    total_preprocessing_times = np.array(total_preprocessing_times)
    total_train_times = np.array(total_train_times, dtype=object)
    total_inference_times = np.array(total_inference_times, dtype=object)
    # print(total_preprocessing_times)
    # print(total_train_times)
    # print(total_inference_times)

    for stage in range(len(args.epoch_setting)):
        if args.dataset.startswith("oag"):
            print(f"Stage: {stage}, Val NDCG: {np.mean(total_best_val_accs[:, stage, 0]):.4f}±"
                f"{np.std(total_best_val_accs[:, stage, 0]):.4f}")
            print(f"Stage: {stage}, Test NDCG: {np.mean(total_best_test_accs[:, stage, 0]):.4f}±"
                f"{np.std(total_best_test_accs[:, stage, 0]):.4f}")
            print(f"Stage: {stage}, Val MRR: {np.mean(total_best_val_accs[:, stage, 1]):.4f}±"
                f"{np.std(total_best_val_accs[:, stage, 1]):.4f}")
            print(f"Stage: {stage}, Test MRR: {np.mean(total_best_test_accs[:, stage, 1]):.4f}±"
                f"{np.std(total_best_test_accs[:, stage, 1]):.4f}")
        else:
            print(f"Stage: {stage}, Val accuracy: {np.mean(total_best_val_accs[:, stage, 0]):.4f}±"
                f"{np.std(total_best_val_accs[:, stage, 0]):.4f}")
            print(f"Stage: {stage}, Test accuracy: {np.mean(total_best_test_accs[:, stage, 0]):.4f}±"
                f"{np.std(total_best_test_accs[:, stage, 0]):.4f}")
        print(f"Stage: {stage}, Preprocessing time: {np.mean(total_preprocessing_times[:, stage]):.4f}±"
            f"{np.std(total_preprocessing_times[:, stage]):.4f}")
        print(f"Stage: {stage}, Training time: {np.hstack(total_train_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_train_times[:, stage]).std():.4f}")
        print(f"Stage: {stage}, Inference time: {np.hstack(total_inference_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_inference_times[:, stage]).std():.4f}")


def define_parser():
    parser = argparse.ArgumentParser(description="Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced Training")
    parser.add_argument("--root", type=str, default="/home/scx/dataset")
    parser.add_argument("--emb-path", type=str, default="/home/scx/NARS/")
    parser.add_argument("--seed", type=int, default=0,
                        help="The start of random seeds.")
    parser.add_argument("--epoch-setting", nargs='+',type=int, default=[10, 10],
                        help="The epoch setting for each stage.")
    parser.add_argument("--warmup-stage", type=int, default=-1,
                        help="(Only for testing) select the stage from which the script starts to train \
                              based on trained files, -1 for cold starting")
    parser.add_argument("--acc-loss", type=str, default="acc",
                        help="Select best model based on validation loss or validation accuracy")
    parser.add_argument("--avoid-features", action="store_true")
    parser.add_argument("--fixed-subsets", action="store_true")
    parser.add_argument("--use-labels", action="store_true")
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--use-norm", action='store_true',
                        help="True=use symmetrically normalized adjacency matrix, False=use row stochastic adjacency matrix")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--relu", type=str, default="relu")
    parser.add_argument("--remove-intermediate-probs", action="store_true")
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--example-subsets-path", type=str, default="/data4/zwt/NARS-main/sample_relation_subsets/examples")
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--K", type=int, default=5,
                        help="The number of hops")
    parser.add_argument("--label-K", type=int, default=3,
                        help="The number of label propagation hops")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="mag")
    parser.add_argument("--model", type=str, default="nars_sagn")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--aggr-gpu", type=int, default=-1,
                        help="Indicate the device ID for graph aggregations.")
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=20000,
                        help="evaluation batch size")
    parser.add_argument("--multihop-layers", type=int, default=2,
                        help="number of multihop MLP layers")
    parser.add_argument("--mlp-layer", type=int, default=2,
                        help="number of MLP layers")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--input-drop", type=float, default=0.,
                        help="dropout on input features")
    parser.add_argument("--attn-drop", type=float, default=0.,
                        help="dropout on attention weights")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="threshold used to generate pseudo hard labels")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")
    parser.add_argument("--probs_dir", type=str, default="../intermediate_outputs",
                        help="The save path of intermediate probability files")
    return parser

if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    main(args)


