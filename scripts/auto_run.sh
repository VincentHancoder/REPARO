NUM_GPUS=8

CONFIG=config/dreamgaussian/furniture_rgbxy_wo_proxy.json
#python evaluation/eval_2d.py --config $CONFIG
 bash scripts/run.sh $CONFIG $NUM_GPUS
# CONFIG=config/triposr/triposr_furniture_rgb_wo_proxy.json
# python evaluation/eval_2d.py --config $CONFIG
# bash scripts/run.sh $CONFIG $NUM_GPUS

