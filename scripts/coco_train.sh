#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=1  tools/train.py -f configs/damoyolo_tinynasL20_T.py -train_ann coco_train_test -val_ann coco_val_test
