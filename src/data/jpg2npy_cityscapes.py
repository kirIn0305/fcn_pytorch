# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='CityScape dataset generater')
    parser.add_argument('--data', '-d', type=str, default="../../data/raw/",
                        metavar='N', help='raw data direcotry')
    parser.add_argument('--cnf', '-c', type=str, default="./config/label_00.csv",
                        metavar='N', help='processed data directory')
    parser.add_argument('--out', '-o', type=str, default="../../data/processed/",
                        metavar='N', help='processed data directory')
    return parser.parse_args()

def color2index(df):
    """color to label index"""
    dict_c2index = {key : value for key, value in zip(df["color"].values, df["trainId"].values)}
    dict_c2index[(0, 0, 0)] = 0 #void class
    return dict_c2index

if __name__ == '__main__':
    args = parse_args()
    img_dir = os.path.join(args.data, 'leftImg8bit')
    lbl_dir = os.path.join(args.data, 'gtFine')
    #output processed data directory
    lbl_name  = os.path.splitext(os.path.basename(args.cnf))[0]
    proc_dir = os.path.join(args.out, lbl_name)
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    df_cnf_lbl = pd.read_csv(args.cnf, index_col=0)
    df_cnf_lbl = df_cnf_lbl[df_cnf_lbl['ignoreInEval'] == False].copy()

    dict_c2index = color2index(df_cnf_lbl)

    tmp_dir = os.listdir(lbl_dir)
    lbl_dirs = {i : os.path.join(lbl_dir, i) for i in tmp_dir if i !='test'} #test is empty.
    city_dirs = {key : os.listdir(value) for key, value in lbl_dirs.items()}

    lbl_train = [os.path.join(lbl_dirs["train"], city) for city in city_dirs["train"]]
    lbl_val = [os.path.join(lbl_dirs["val"], city) for city in city_dirs["val"]]
    img_train = [city_dir.replace('gtFine', 'leftImg8bit') for city_dir in lbl_train]
    img_val = [city_dir.replace('gtFine', 'leftImg8bit') for city_dir in lbl_val]

    for city_dir in lbl_train:
        print(city_dir)
        clbls_fnames = [os.path.join(city_dir, i) for i in os.listdir(city_dir) if "color" in i]
        for c_fname in clbls_fnames:
            c_lbl = np.array(Image.open(c_fname).convert("RGB"))
            c_lbl_flat = [str(tuple(i)) for i in c_lbl.reshape([-1, 3])]
            c_lbl_convert = np.array([dict_c2index[i] if str(i) in list(dict_c2index.keys()) else 19 for i in c_lbl_flat]).reshape(c_lbl.shape[:2])

