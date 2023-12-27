import argparse
import sys
from yacs.config import CfgNode as CN

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--process_dataset', help='whether to process data')
args = parser.parse_args()


with open(args.config, 'r') as fid:
    FLAGS = CN.load_cfg(fid)
