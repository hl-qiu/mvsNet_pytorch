#!/usr/bin/env bash

DTU_TESTING="/data/jh/code/tf/dense/"
CKPT_FILE="./checkpoints/d192/model_000004.ckpt"
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test_t.txt --loadckpt $CKPT_FILE $@
