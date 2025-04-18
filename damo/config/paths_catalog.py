# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'train_coco_04072025': {
            'img_dir': 'training_04072025/images/default',
            'ann_file': 'training_04072025/annotations/train.json'
            },
        'val_coco_04072025': {
            'img_dir': 'training_04072025/images/default',
            'ann_file': 'training_04072025/annotations/val.json'
            },
        'train_coco_04182025': {
            'img_dir': 'face_training_04182025/images/default',
            'ann_file': 'face_training_04182025/annotations/train.json'
            },
        'val_coco_04182025': {
            'img_dir': 'face_training_04182025/images/default',
            'ann_file': 'face_training_04182025/annotations/val.json'
            } 
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
