import os
# gpu index for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ast import parse
import jittor as jt
# from train import Trainer
# from model import NerfNetworks, HuberLoss
from tqdm import tqdm
# from utils.dataset import  NerfDataset
import argparse
import numpy as np
from jnerf.runner import Runner 
from jnerf.utils.config import init_cfg
# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1


def main():
    assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="test",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )
    
    args = parser.parse_args()

    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()
    runner.test(True)
    
if __name__ == "__main__":
    main()