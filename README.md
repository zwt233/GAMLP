### ogbn-products

```
python main.py --use-rdd --method R_GAMLP_RDD --stages 400 300 300 300 --train-num-epochs 0 0 0 0 --threshold 0.85 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --gpu 6 --eval 10 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 4 --root  /data2/zwt/ --gpu 0 --seed 0 --bns --gama 0.1
```

#### ogbn-mag

```
python main.py --use-rdd --method JK_GAMLP_RDD --stages 250 200 200 200 --train-num-epochs 0 0 0 0 --threshold 0.4 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 300 --n-layers-1 4 --n-layers-2 4 --label-num-hops 3 --seed 0 --gpu 1 --bns --gama 10 --use-relation-subsets ./data/mag --emb_path ./data/
```

 #### ogbn-100Mpapers

```
python main.py --use-rdd --method R_GAMLP_RDD --stages 100 150 150 150 --train-num-epochs 0 0 0 0 --threshold 0 --input-drop 0 --att-drop 0 --label-drop 0 --dropout 0.5 --pre-process --dataset ogbn-papers100M --num-runs 3 --eval 1 --act sigmoid --batch 5000 --patience 300 --n-layers-2 6 --label-num-hops 9 --seed 0 --gpu 1  --root /data2/zwt/ --num-hops 6 --hidden 1024 --bns --temp 0.001
```



| Method    | ogbn-products validation | ogbn-products test | ogbn-papers100M validation | ogbn-papers100M test | ogbn-mag validation | ogbn-mag test |
| --------- | ------------------------ | ------------------ | -------------------------- | -------------------- | -------------------------- | -------------------- |
| GAMLP+RDD | **93.24±0.05\%**         | **84.59±0.10\%**   | **71.59±0.05\%** | **68.25±0.11\%**  |**57.02±0.41\%**|**55.90±0.27\%**|

