### ogbn-products

```
python main.py --use-rdd --method R_GAMLP_RDD --stages 400 200 200 200 --train-num-epochs 0 0 0 0 --threshold 0.85 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --gpu 6 --eval 10 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 4 --root  /data2/zwt/ --gpu 0 --seed 0 --bns
```

seed 0 9324 8467

seed 1 9325 8453

seed 2 9311 8452

seed 3 9319 8457

seed 4 9321 8459

seed 5 9314 8453

seed 6 9320 8449

seed 7 9323 8460

seed 8 9319 8454

seed 9 9326 8451

#### ogbn-mag

```
python main.py --use-rdd --method JK_GAMLP_RDD --stages 500 200 200 200 --train-num-epochs 0 0 0 0 --threshold 0.4 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 300 --n-layers-1 6 --n-layers-2 4 --label-num-hops 3 --seed 0 --gpu 1 --bns
```

seed 0 5762 5662

seed 1 0.5766,0.5656

seed 2 5738 5611

seed 3 5754 5644

seed 4 5750 5626

seed 5 5719 5613

seed 6 0.5787,0.5635

seed 7 5767 5615

seed 8 5745 5618

seed 9 5732 5585

 #### ogbn-100Mpapers

```
python main.py --use-rdd --method JK_GAMLP_RDD --stages 300 300 300 --train-num-epochs 100 50 50  --threshold 0.6 --input-drop 0 --att-drop 0.5 --label-drop 0 --pre-process --dataset ogbn-papers100M --num-runs 3 --eval 1 --act sigmoid --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 4 --label-num-hops 9 --seed 0 --gpu 1 --bns --pre-dropout --act sigmoid --num-hops 16 --hidden 1024
```
