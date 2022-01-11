import sys
import os

# root_path = os.path.abspath(__file__)
# root_path = '/'.join(root_path.split('/')[:-2])

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
import argparse
from train.fish_train import fish

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="scuba diving fish recognition")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(root_path,"model_weight"), help="as named")
    parser.add_argument("--pretrain_model_path", type=str,
                        default=os.path.join(root_path,"model_weight"), help="as named")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="as named")
    parser.add_argument("--lr", type=float,
                        default=3e-4, help="as named")
    parser.add_argument("--max_lr", type=float,
                        default=3e-4, help="as named")
    parser.add_argument("--min_lr", type=float,
                        default=3e-6, help="as named")
    parser.add_argument("--epoch", type=int,
                        default=80, help="as named")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(root_path,"fish_photos"), help="as named")
    parser.add_argument("--cuda_index", type=int,
                        default=0, help="as named")
    parser.add_argument("--model_name", type=str,
                        default="resnet34", help="as named")
    args = parser.parse_args()
    fish(args)