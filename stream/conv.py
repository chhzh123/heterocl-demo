import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report
import hlib

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode="csyn"):
        hcl.init()
        A = hcl.placeholder((1, 1, 16, 16), "A")
        w = hcl.placeholder((1, 3, 3, 3), "w")
        def kernel(A, w):
            # B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            B = hlib.op.nn.conv2d_nchw(A, w, padding=[1,1], name="B")
            return B
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, w], kernel)
        s.to([A, w], target.xcel)
        s.to(kernel.B, target.host)
        target.config(compile="vivado_hls", mode=target_mode)
        f = hcl.build(s, target=target)

        np_A = np.random.randint(10, size=(1,1,16,16))
        np_w = np.random.randint(10, size=(1,3,3,3))
        np_B = np.zeros((1,3,16,16))

        hcl_A = hcl.asarray(np_A)
        hcl_w = hcl.asarray(np_w)
        hcl_B = hcl.asarray(np_B)
        f(hcl_A, hcl_w, hcl_B)

    test_hls()

if __name__ == "__main__":
    test_vivado_hls()