#!/bin/bash
#In most situations, only need to alter the name of the directory under "datasets"
ANNOTATIONS_PATH="datasets/eye_training_04182025/annotations.xml"
ORIGINAL_IMAGES="datasets/eye_training_04182025/images/default"
NEW_TRAINING_SET="datasets/eye_training_04182025/training"
#Shouldn't need to change this 
VALIDATION_SET="datasets/bag_eyevalidation_04292025"
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/eyeRegression.py -a ${ANNOTATIONS_PATH} -i ${ORIGINAL_IMAGES} -ti ${NEW_TRAINING_SET} -va ${VALIDATION_SET}