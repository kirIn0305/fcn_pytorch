# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='CityScape dataset generater')
    parser.add_argument('--data', '-d', type=str, default="../../data/raw/",
                        metavar='N', help='raw data direcotry')
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

    df_cnf_lbl = pd.read_csv('./config/config_label.csv', index_col=0)

    #ignore in Eval data parse
    df_cnf_lbl = df_cnf_lbl[df_cnf_lbl['ignoreInEval'] == False].copy()

    dict_c2index = color2index(df_cnf_lbl)
