'''
Author: sunflower-you m13119160579@163.com
Date: 2024-05-06 22:06:44
LastEditors: sunflower-you m13119160579@163.com
LastEditTime: 2024-05-06 22:58:59
FilePath: /tvm_learning/tvmv0.15.dev0/tvm/tests/python/relay/test_swiglu.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import tensorflow as tf
import tvm
from tvm.contrib import graph_executor  
import tvm.testing
from tvm import relay
import time  


def tf_swiglu(data, gate_weight, up_weight, gate_bias_flag, up_bias_flag):
    output = tf.nn.silu(tf.matmul(data, gate_weight)) * (tf.matmul(data, up_weight))
    return output


def verify_op(datashape, gate_weightshape, up_weightshape, gate_biasshape, up_biasshape, output_max, bias_term):
        data = relay.var("data", relay.TensorType(datashape, "float32"))
        gate_weight = relay.var("gate_weight", relay.TensorType(gate_weightshape, "float32"))
        up_weight = relay.var("up_weight", relay.TensorType(up_weightshape, "float32"))
        gate_bias = relay.var("gate_bias", relay.TensorType(gate_biasshape, "float32"))
        up_bias = relay.var("up_bias", relay.TensorType(up_biasshape, "float32"))

        op = relay.op.swiglu(data, gate_weight, up_weight, gate_bias, up_bias, output_max, bias_term)
        func = relay.Function([data, gate_weight, up_weight, gate_bias, up_bias], op)
        module = tvm.IRModule.from_expr(func)

        target = tvm.target.Target(target="llvm")
        device = tvm.device(target.kind.name, 0)
        target = "llvm"
        lib = relay.build_module.build(module, target, params=None)
        module = graph_executor.GraphModule(lib["default"](device))

        data = np.random.uniform(0, 1, size=datashape).astype("float32")
        gate_weight = np.random.randint(-20, 20, size=gate_weightshape).astype("float32")
        up_weight = np.random.randint(-20, 20, size=up_weightshape).astype("float32")
        gate_bias = np.random.randint(5, 10, size=gate_biasshape).astype("float32")
        up_bias = np.random.randint(5, 10, size=up_biasshape).astype("float32")

        data_re = data.copy()
        data_np = data.copy()
        gate_weight = gate_weight.copy()
        up_weight = up_weight.copy()
        gate_bias = gate_bias.copy()
        up_bias = up_bias.copy() 
        # M_re = M.copy()
        # M_np = M.copy()
        # print("input arr:", arr)  
        # print("input indices:", indices)  
        # print("input values:", values)  

        module.set_input("data", tvm.nd.array(data_re))
        module.set_input("gate_weight", tvm.nd.array(gate_weight))
        module.set_input("up_weight", tvm.nd.array(up_weight))
        module.set_input("gate_bias", tvm.nd.array(gate_bias))
        module.set_input("up_bias", tvm.nd.array(up_bias))

        start_time = time.time()  
        module.run()
        end_time = time.time()  
        print(f"relay time: {(end_time - start_time) * 1000:.2f} ms") 
        output = module.get_output(0).asnumpy()
        print("output:")
        print(output)

        start_time = time.time()  
        ref_res = tf_swiglu(data_np, gate_weight, up_weight, gate_bias, up_bias)
        end_time = time.time()  
        print(f"np time: {(end_time - start_time) * 1000:.2f} ms") 

        print("ref_res")
        print(ref_res)
        tvm.testing.assert_allclose(output, ref_res, rtol=1e-3)
        print(" is right!")


def test_quant_batch_matmul():
    n = 3
    a = 4
    b = 5
    c = 7
    input_max = np.random.uniform(1, 1.2)
    output_max = np.random.uniform(1, 2.0)
    bias_term = True
    verify_op((n,a,b,) ,(1,c,b,) , (1,c,b,), (c,), (c,))

test_quant_batch_matmul()
