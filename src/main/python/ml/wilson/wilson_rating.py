# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/10/29
version : 
"""
import numpy as np
import pandas as pd


def wilson_lower_interval(p, n, z):
    p = p * 1.0
    if (p <= 0 or n == 0):
        return 0
    if (n != 0):
        p = (p + pow(z, 2) * 1.0 / (2 * n) - z * np.sqrt(
            p * (1 - p) / n + pow(z, 2) * 1.0 / (4 * pow(n, 2)))) / (1 + 1.0 * pow(z, 2) / n)
        return p
    return 0


def wilson_udf(p_and_n, z):
    p, n = p_and_n.split(',')
    p = float(p)
    n = float(n)
    # n = pow(np.e, float(n)) - 1
    return wilson_lower_interval(p, n, z)


def wilson(df, p, n, z):
    df['p_and_n'] = df[p].apply(str) + "," + df[n].apply(str)
    df.insert(df.loc[:, :p].shape[1], 'wilson_score', df['p_and_n'].apply(lambda x: wilson_udf(x, z)))
    del df['p_and_n']
    return df


if __name__ == '__main__':
    df = pd.read_csv("E:/learning-to-rank/resources/sh_pdjc.csv", header="infer")
    df['dist_reverse'] = 1 / df['dist']
    df = wilson(df, 'desc', 'dist_reverse', 1.96)
    df.to_csv('E:/df.csv', index=False)
