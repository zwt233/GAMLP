cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../src/main.py \
    --dataset acm \
    --num-runs 10 \
    --K 2 \
    --num-hidden 64 \
    --lr 0.007 \
    --dropout 0.7 \
    --acc-loss acc \
    --batch-size 10000  \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 1 \
    --epoch-setting 300 200 200 \
    --attn-drop 0.3 \
    --input-drop 0.7 \
    --sample-size 2 \
    --threshold 0.3 \
    --model nars_sagn \
    --multihop-layer 2 \
    --mlp-layer 2 \
    --relu prelu \
    --no-batch-norm \
    --label-K 3 \
    --use-labels