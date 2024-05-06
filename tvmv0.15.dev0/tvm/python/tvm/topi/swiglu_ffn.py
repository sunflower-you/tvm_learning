'''
Author: sunflower-you m13119160579@163.com
Date: 2024-05-06 09:44:07
LastEditors: sunflower-you m13119160579@163.com
LastEditTime: 2024-05-06 10:00:15
FilePath: /tvm_learning/tvmv0.15.dev0/tvm/python/tvm/topi/swiglu_ffn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import tvm
import tvm.topi
from ..te import hybrid
from .math import *
from .broadcast import *
from .transform import *


def swiglu_ffn(data, gate_weight, up_weight, gate_bias,
               up_bias, gate_bias_flag, up_bias_flag):
    if not gate_bias_flag:
        gate_bias = None
    if not up_bias_flag:
        up_bias = None

    output = tvm.topi.silu(matmul(data, gate_weight) + gate_bias) * (matmul(data, up_weight) + up_bias)
    return output