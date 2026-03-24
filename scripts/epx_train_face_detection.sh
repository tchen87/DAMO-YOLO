#!/bin/bash
#In most situations, only need to alter the name of the directory under "datasets"
INSTANCES_PATH="datasets/fullDataset_10222025_coco/annotations/instances_default.json"
TRAIN_JSON_PATH="datasets/fullDataset_10222025_coco/annotations/train.json"
VAL_JSON_PATH="datasets/fullDataset_10222025_coco/annotations/val.json"
IMG_DIR="datasets/fullDataset_10222025_coco/images/default"
#NOTE: These must start with "coco". Good to add a date to the name
TRAIN_NAME="coco_train_10222025"
VAL_NAME="coco_val_10222025"
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/cocosplit.py --having-annotations -s 0.8 ${INSTANCES_PATH} ${TRAIN_JSON_PATH} ${VAL_JSON_PATH}
python tools/AddDatasetToConfigFiles.py -ti ${IMG_DIR} -ta ${TRAIN_JSON_PATH} -tn ${TRAIN_NAME} -vi ${IMG_DIR} -va ${VAL_JSON_PATH} -vn ${VAL_NAME}
python -m torch.distributed.launch --nproc_per_node=1  tools/train.py -f configs/damoyolo_tinynasL20_T.py --train_ann ${TRAIN_NAME} --val_ann ${VAL_NAME}