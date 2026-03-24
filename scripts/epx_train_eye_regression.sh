#!/bin/bash
#In most situations, only need to alter the name of the directory under "datasets"
ANNOTATIONS_PATH="datasets/fullDataset_10222025_cvat/annotations.xml"
ORIGINAL_IMAGES="datasets/fullDataset_10222025_cvat/images/default"
NEW_TRAINING_SET="datasets/fullDataset_10222025_cvat/training"
#Shouldn't need to change this 
VALIDATION_SET="datasets/bag_eyevalidation_04292025"
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/eyeRegression.py -a ${ANNOTATIONS_PATH} -i ${ORIGINAL_IMAGES} -ti ${NEW_TRAINING_SET} -va ${VALIDATION_SET}