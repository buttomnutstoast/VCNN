#!/usr/bin/env python
import sys
sys.path.append('pyutils/visual-concepts')
import os

import argparse, pprint

import sg_utils as utils
import preprocess
import coco_voc
from test_model import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Script for testing word detection models.'
        )

    parser.add_argument('--vocab_file', dest='vocab_file',
        help='vocabulary to train for',
        default='dataset/mscoco/vocabs/vocab_train.pkl',
        type=str)

    parser.add_argument('--det_file', dest='det_file',
        help='file with prediction scores are saved',
        default='checkpoint/mscoco/prediction.h5',
        type=str)

    parser.add_argument('--map_file', dest='map_file',
        help='file of mapping from detection category to caption label',
        default='dataset/mscoco/vocabs/coco2vocab_manual_mapping.txt',
        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    imdb = coco_voc.coco_voc('test')
    vocab = utils.load_variables(args.vocab_file)
    gt_label = preprocess.get_vocab_counts(
        imdb.image_index,
        imdb.coco_caption_data,
        5,
        vocab
        )
    det_file = args.det_file
    det_dir = os.path.dirname(det_file) # get root dir of det_file

    eval_file = os.path.join(det_dir, imdb.name + '_eval.pkl')
    benchmark(imdb, vocab, gt_label, 5, det_file, eval_file=eval_file)

    map_file = args.map_file
    gt_label_det = preprocess.get_vocab_counts_det(
        imdb.image_index,
        imdb.coco_instances_data,
        map_file,
        vocab
        )
    eval_file = os.path.join(det_dir, imdb.name + '_eval_det.pkl')
    benchmark_det(imdb, vocab, gt_label_det, map_file, det_file, eval_file=eval_file)
