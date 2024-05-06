'''
Author: sunflower-you m13119160579@163.com
Date: 2024-05-07 00:32:09
LastEditors: sunflower-you m13119160579@163.com
LastEditTime: 2024-05-07 00:40:55
FilePath: /tvm_learning/tvmv0.15.dev0/tvm/python/tvm/relay/quantize/range_estimators.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from calendar import c
import numpy as np
import copy
from scipy.optimize import minimize_scalar
import logging
from enum import Enum
import multiprocessing as mp
import time
import ctypes
import numba

from . import _quantize

def get_pointer(arr, ctypes_type):
    """
    Get the numpy pointer which pass to c++ end.
    """
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes_type))
    return ctypes.cast(ptr, ctypes.c_void_p)

class RangeEstimatorBase(object):
    def __init__(self, per_channel=False, quantizer=None, axis=None, n_groups=None, *args,
                 **kwargs):
        self.current_xmin = None
        self.current_xmax = None
        self.per_channel = per_channel
        self.quantizer = quantizer
        self.axis = axis
        self.n_groups = n_groups

        self.per_group_range_estimation = False
        self.ranges = None

        # Works for activation, since step 1 already decide the activation's min max
        self.max_pos_thr_out = None
        self.max_neg_thr_out = None
        self.one_sided_dist = None
        self.data_tmp = None
    
    def calibrate(self, x):
        """
        Accepts an input tensor, updates the current estimates of x_min and x_max
        and returns them.
        Parameters
        ----------
        x: Input tensor

        Returns
        -------
        self.current_xmin: tensor
        self.current_xmax: tensor
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the range estimator.
        """
        self.current_xmin = None
        self.current_xmax = None
    
    def set_min_max(self, min_val, max_val):
        self.max_pos_thr_out = max_val
        self.max_neg_thr_out = min_val
        self.one_sided_dist = bool(self.max_neg_thr_out >= 0)


class PercentileEstimator(RangeEstimatorBase):
    def __init__(self, bins=8001,percentile=99.99,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins
        self.percentile = percentile
        
    def get_amx_percentile(self,x,percentile,axis=None):
        calib_hist, calib_bin_edges = np.histogram(x, bins=self.bins)
        if percentile < 0 or percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        total = calib_hist.sum()
        cdf = np.cumsum(calib_hist / total,axis)
        idxmax = np.searchsorted(cdf, percentile / 100)
        idxmin = np.searchsorted(cdf, (100-percentile) / 100)
        
        calib_amax = calib_bin_edges[idxmax]
        calib_amin = calib_bin_edges[idxmin]
        
        return calib_amax,calib_amin
    
    def calibrate(self, x):
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.reshape(x.shape[0], -1)
            x_max,x_min= self.get_amx_percentile(x_flattened,self.percentile,-1) # type: ignore
        else:
            x_max,x_min = self.get_amx_percentile(x,self.percentile,None) # type: ignore
        
        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = np.minimum(self.current_xmin, x_min)
            self.current_xmax = np.maximum(self.current_xmax, x_max)

        return self.current_xmin, self.current_xmax