#!/usr/bin/env python3

import argparse
import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from scraper.scrape import scraper
import json


def scrap_main(args):
    with open(args.fish_path, 'r') as file:
        names = list(json.load(file))
    scraper(names, os.path.join(root_path, "fish_photos"), args.limit)


# from scraper.scrap_main import fetch_go
#
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='爬取鱼的训练数据集')
    parser.add_argument("--limit", type=int,
                        default=20, help="每个种类需要爬取的图片的数量")
    parser.add_argument("--fish_path", type=str,
                        default="fish.json", help="需要的种类的json路径")
    args = parser.parse_args()
    scrap_main(args)

# fetch_go(args.num)
