import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report
import hlib.op.bnn as bnn

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode="csyn"):
        input_image = hcl.placeholder((1, 1, 16, 16), "input_image",dtype=hcl.UInt(1))
        w_conv1 = hcl.placeholder((1, 16, 3, 3), "w_conv1",dtype=hcl.UInt(1))
        bn_t1 = hcl.placeholder((16, 16, 16), "bn_t1",dtype=hcl.Float())
        def kernel(input_image, w_conv1):
            conv1 = bnn.conv2d_nchw(input_image, w_conv1,
                                    padding=[1,1], name="conv1", out_dtype=hcl.Int(8))
            return conv1
            bn1 = bnn.batch_norm_threshold(conv1, bn_t1, name="bn1")
            return bn1
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([input_image, w_conv1], kernel)
        s.to([input_image, w_conv1], target.xcel)
        s.to(kernel.conv1, target.host)
        target.config(compile="vivado_hls", mode=target_mode)
        # f = hcl.build(s, target=None)
        f = hcl.build(s, target)

        np_input_image = np.random.randint(0, 2, size=(1,1,16,16))
        np_w_conv1 = np.random.randint(0, 2, size=(1,16,3,3))
        np_bn_t1 = np.random.randint(0, 10, size=(16,16,16))
        np_out = np.zeros((1,16,16,16))

        hcl_input_image = hcl.asarray(np_input_image,dtype=hcl.UInt(1))
        hcl_w_conv1 = hcl.asarray(np_w_conv1,dtype=hcl.UInt(1))
        hcl_bn_t1 = hcl.asarray(np_bn_t1,dtype=hcl.Float())
        hcl_out = hcl.asarray(np_out,dtype=hcl.Int(8))
        f(hcl_input_image, hcl_w_conv1, hcl_out)

    test_hls()

if __name__ == "__main__":
    test_vivado_hls()