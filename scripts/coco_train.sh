#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1  tools/train.py -f configs/damoyolo_tinynasL20_T.py
