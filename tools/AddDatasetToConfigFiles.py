

import argparse
import ast

from torch._inductor.ir import NoneAsConstantBuffer
import astor
from loguru import logger

def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')

    parser.add_argument('-ti',
                        '--training_imgdir',
                        default=None,
                        type=str,
                        help='Path to directory containing training images',)
    parser.add_argument('-ta',
                        '--training_annotationfile',
                        default=None,
                        type=str,
                        help='Path to training annotation json file')
    parser.add_argument('-vi',
                        '--validation_imgdir',
                        default=None,
                        type=str,
                        help='Path to directory containing validation images',)
    parser.add_argument('-va',
                        '--validation_annotationfile',
                        default=None,
                        type=str,
                        help='Path to validation annotation json file')
    parser.add_argument('-tn',
                        '--training_dataset_name',
                        default=None,
                        type=str,
                        help='Name to give training dataset')
    parser.add_argument('-vn',
                        '--validation_dataset_name',
                        default=None,
                        type=str,
                        help='Name to give validation dataset')
    return parser



def add_datasets_to_file(file_path, new_entries):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    insert_index = None
    for i, line in enumerate(lines):
        if 'DATASETS = {' in line:
            insert_index = i + 1  # Insert immediately after opening line
            break

    if insert_index is None:
        raise RuntimeError("DATASETS dictionary not found or improperly formatted.")

    # Generate new entry lines
    new_lines = []
    for entry in reversed(new_entries):  # reverse to preserve order
        new_lines = [
            f"        '{entry['name']}': {{\n",
            f"            'img_dir': '{entry['img_dir']}',\n",
            f"            'ann_file': '{entry['ann_file']}'\n",
            f"            }},\n"
        ] + new_lines

    # Insert new entries
    lines[insert_index:insert_index] = new_lines

    # Write updated lines back to file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Inserted {len(new_entries)} new entries at the beginning of DATASETS in {file_path}")


def main():
    args = make_parser().parse_args()
    training_imgdir = args.training_imgdir
    training_annotationfile = args.training_annotationfile
    validation_imgdir = args.validation_imgdir
    validation_annotationfile = args.validation_annotationfile
    training_dataset_name = args.training_dataset_name
    validation_dataset_name = args.validation_dataset_name

    if training_imgdir == None :
        logger.debug("Training image directory path required")
        return
    if training_annotationfile == None :
        logger.debug("Training annotation file path required")
        return
    if training_dataset_name == None :
        logger.debug("Training dataset name required")
        return
    if validation_imgdir == None :
        logger.debug("Validation image directory path required")
        return
    if validation_annotationfile == None :
        logger.debug("Validation annotation file path required")
        return
    if validation_dataset_name == None :
        logger.debug("Validation dataset name required")
        return

    paths_catalog_path = "damo/config/paths_catalog.py"
    # Define new dataset entries
    new_entries = [
        {
            "name":training_dataset_name,
            "img_dir":training_imgdir,
            "ann_file":training_annotationfile
        },
        {
            "name": validation_dataset_name,
            "img_dir": validation_imgdir,
            "ann_file": validation_annotationfile
        }
    ]

    add_datasets_to_file(paths_catalog_path, new_entries)



if __name__ == '__main__':
    main()
