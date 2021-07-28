cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../src/main.py \
    --dataset mag \
    --num-runs 10 \
    --K 5 \
    --num-hidden 512 \
    --lr 0.001 \
    --dropout 0.5 \
    --batch-size 10000  \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 10 \
    --root /home/scx/dataset \
    --epoch-setting 500 200 150 \
    --attn-drop 0. \
    --input-drop 0. \
    --sample-size 8 \
    --threshold 0.4 \
    --model nars_sagn \
    --multihop-layers 2 \
    --mlp-layer 2 \
    --label-K 3 \
    --fixed-subsets \
    --use-labels 
    # --relu prelu \
    # --no-batch-norm \
    