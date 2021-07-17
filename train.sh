#!/bin/bash

source ./set_env.sh
which python

nohup python -u multigpu_train.py \
    --gpu_list=0,1 \
    --input_size=512 \
    --batch_size_per_gpu=14 \
    --checkpoint_dir=./checkpoints/ \
    --train_img_list_path=/input/workspace/zuming.hzm/data/icdar_data/rctw17_mlt17/img_list.txt \
    --train_xml_list_path=/input/workspace/zuming.hzm/data/icdar_data/rctw17_mlt17/xml_list.txt \
    --geometry=QUAD \
    --learning_rate=0.0001 \
    --num_readers=24 \
    --pretrained_model_path=./pretrained_model/resnet_v1_50.ckpt \
    > log/train.log 2>&1 &