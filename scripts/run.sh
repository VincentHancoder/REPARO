CONFIG=$1
NUM_GPUS=$2

echo "$CONFIG"
for ((IDX=0; IDX<NUM_GPUS; IDX++))
do
    echo "$IDX"
    CUDA_VISIBLE_DEVICES=$IDX python experiments/furniture/optim_reparo.py --config $CONFIG --chunk_num $NUM_GPUS --chunk_id $IDX &
done

wait

# eval
python evaluation/eval_2d.py --config $CONFIG