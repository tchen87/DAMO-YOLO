# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'first_batch_train_coco': {
            'img_dir': 'firstBatch/images/default',
            'ann_file': 'firstBatch/annotations/train.json'
            },
        'first_batch_val_coco': {
            'img_dir': 'firstBatch/images/default',
            'ann_file': 'firstBatch/annotations/val.json'
            },
        'second_batch_train_coco': {
            'img_dir': 'secondBatchAdded/images/default',
            'ann_file': 'secondBatchAdded/annotations/train.json'
            },
        'second_batch_val_coco': {
            'img_dir': 'secondBatchAdded/images/default',
            'ann_file': 'secondBatchAdded/annotations/val.json'
            },
        'train_coco_03012025': {
            'img_dir': 'fullDataset03012025/images/default',
            'ann_file': 'fullDataset03012025/annotations/train.json'
            },
        'val_coco_03012025': {
            'img_dir': 'fullDataset03012025/images/default',
            'ann_file': 'fullDataset03012025/annotations/val.json'
            },
        'train_coco_03132025': {
            'img_dir': 'updatedLabels_03132025/images/default',
            'ann_file': 'updatedLabels_03132025/annotations/train.json'
            },
        'val_coco_03132025': {
            'img_dir': 'updatedLabels_03132025/images/default',
            'ann_file': 'updatedLabels_03132025/annotations/val.json'
            },   
        'train_coco_03282025': {
            'img_dir': 'training_03282025/images/default',
            'ann_file': 'training_03282025/annotations/train.json'
            },
        'val_coco_03282025': {
            'img_dir': 'training_03282025/images/default',
            'ann_file': 'training_03282025/annotations/val.json'
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
