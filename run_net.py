import os
# gpu index for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        default="train",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--save_test_result_dir",
        default=os.path.dirname(os.path.realpath(__file__)),
        type=str,
    )
    
    args = parser.parse_args()

    assert args.task in ["train","test","render", "fine_tune"],f"{args.task} not support, please choose [train, test, render]"
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()

    if args.task == "train":
        runner.train()
    elif args.task == "test":
        runner.test(True)
    elif args.task == "render":
        runner.get_video()
        # runner.render(True, args.save_dir)
    elif args.task == 'fine_tune':
        runner.fine_tune()
    
if __name__ == "__main__":
    main()