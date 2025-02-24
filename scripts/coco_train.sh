#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL18_Nm.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL18_Ns.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL20_N.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL20_Nl.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL20_T.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL25_S.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL35_M.py
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL45_L.py
