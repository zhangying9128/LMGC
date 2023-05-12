# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:40:27 2021

@author: Mac
"""

import numpy as np

def make_span_perm( perm,  word_idx,  n):
    ans = np.zeros(n, dtype=np.int64)

    g = 0
    for i in range(len(word_idx) - 1):
        start = word_idx[perm[i]]
        end = word_idx[perm[i] + 1]
        for j in range(start, end):
            ans[g] = j
            g = g + 1
    return ans
