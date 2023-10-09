#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd



class dscovr_preprocessor:
    """
    does preprocessing for faraday cup vars; call like
    dp_inst = dp()
    subset = dp.drop_nan_threshold(data, threshold=0.3)
    subset = dp.select_across_k_cups(subset, k=5)
    """
    
    def __init__(self):
        pass
    
    
    def get_fc_vars(self, data):
        return [ elem for elem in list(data) if 'FC' in elem ]
    
    
    
    def drop_nan_threshold(self, data, threshold=None):
        if threshold is None: threshold = 0.2
        
        fc_vars = self.get_fc_vars(data)
        other_vars = np.setdiff1d( list(data), fc_vars ).tolist()
        fc_dat = data[fc_vars]

        feature_nan_fracs = fc_dat.isna().sum(axis=0) / fc_dat.shape[0]
        print('nan dist:',feature_nan_fracs)
        
        sort_inds = np.argsort(feature_nan_fracs).values
        feat_inds_to_drop = sort_inds[::-1][ :int( sort_inds.shape[0] * threshold ) ]
        feats_to_drop = np.array(fc_vars)[feat_inds_to_drop]
        print('dropped',len(feats_to_drop),'features:',feats_to_drop)
        
        # extract vars that weren't dropped ...
        vars_to_keep = np.setdiff1d( fc_vars, feats_to_drop ).tolist()
        # ... and make sure that they're in the original FC sorted order!
        vars_to_keep = np.array(vars_to_keep)[ np.argsort( [ int(elem[2:]) for elem in vars_to_keep ] ) ]
        
        # rejoin original vars and kept features
        return pd.concat( [ data[other_vars], fc_dat[vars_to_keep] ], axis=1 )

    
    
    
    def select_across_k_cups(self, data, k=None):
        if k is None: k = 5
        
        # get fc vars for data provided
        fc_vars = self.get_fc_vars(data)
        other_vars = np.setdiff1d( list(data), fc_vars ).tolist()
        
        # keep only FC vars for those in blocks of length k - ignore remainder
        feat_inds_to_keep = len(fc_vars) - (len(fc_vars) % k)
        feats_to_keep_k_blocks = np.array(fc_vars)[ :feat_inds_to_keep ].tolist()
        
        # extract first feature of every k-length block
        feat_inds_to_keep_per_block = np.arange(0, feat_inds_to_keep, step=k)
        feats_to_keep = np.array(feats_to_keep_k_blocks)[ feat_inds_to_keep_per_block ].tolist()
        
        #return data[feats_to_keep]
        return pd.concat( [ data[other_vars], data[feats_to_keep] ], axis=1 )

    def high_value_cutoff(self, data, cutoff):
        fc_vars = self.get_fc_vars(data)
        other_vars = np.setdiff1d( list(data), fc_vars ).tolist()
        fc_dat = data[fc_vars]

        fc_dat.values = np.clip(fc_dat.values, 0, cutoff)

        return pd.concat( [ data[other_vars], fc_dat ], axis=1 )