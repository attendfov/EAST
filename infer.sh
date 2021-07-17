#!/bin/bash

source ./set_env.sh

python -u infer.py \
    --gpu_list=0 \
    --test_data_dir=/gruntdata/DL_dataset/zuming.hzm/data/scene_text/security_scene_text_19/processed_data/test/rotate_data \
    --checkpoint_dir=./checkpoints/ \
    --max_side_len=512 \
    --output_dir=./results/security_scene_text_19_99991_permute_resize_512_refine \
    --save_score_map=True \
    --enable_box_refinement=True \
    | tee log/infer.log 2>&1