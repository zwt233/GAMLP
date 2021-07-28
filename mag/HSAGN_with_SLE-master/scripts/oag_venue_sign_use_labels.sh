cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../src/main.py \
    --dataset oag_venue \
    --remove-intermediate-probs \
    --num-runs 10 \
    --K 3 \
    --num-hidden 256 \
    --lr 0.001 \
    --dropout 0.5 \
    --batch-size 1000  \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 10 \
    --root /home/scx/NARS/oag_dataset \
    --epoch-setting 200 100 100 \
    --attn-drop 0. \
    --input-drop 0. \
    --sample-size 8 \
    --threshold 0.2 \
    --model nars_sign \
    --multihop-layers 2 \
    --mlp-layer 2 \
    --label-K 1 \
    --relu prelu \
    --no-batch-norm \
    --use-labels
