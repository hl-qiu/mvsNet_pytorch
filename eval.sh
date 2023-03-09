#!/usr/bin/env bash
#DTU_TESTING="data/mvs_testing/dtu/"
DTU_TESTING="data/self_made/"
CKPT_FILE="./checkpoints/d192/model_000014.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/self_made_testlist/test.txt --loadckpt $CKPT_FILE $@

