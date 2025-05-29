#!/bin/bash
INSTANCES_PATH="datasets/exampleName/annotations/instances_default.json"
TRAIN_JSON_PATH="datasets/exampleName/annotations/train.json"
VAL_JSON_PATH="datasets/exampleName/annotations/val.json"
IMG_DIR="datasets/exampleName/images/default"
#NOTE: These must start with "coco" 
TRAIN_NAME="coco_train_example"
VAL_NAME="coco_val_example"
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/cocosplit.py --having-annotations -s 0.8 ${INSTANCES_PATH} ${TRAIN_JSON_PATH} ${VAL_JSON_PATH}
python tools/AddDatasetToConfigFiles.py -ti ${IMG_DIR} -ta ${TRAIN_JSON_PATH} -tn ${TRAIN_NAME} -vi ${IMG_DIR} -va ${VAL_JSON_PATH} -vn ${VAL_NAME}
python -m torch.distributed.launch --nproc_per_node=1  tools/train.py -f configs/damoyolo_tinynasL20_T.py --train_ann ${TRAIN_NAME} --val_ann ${VAL_NAME}