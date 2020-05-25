import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report
import hlib

def partition_test():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(8))
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=hcl.UInt(8))
        return B

    target = hcl.platform.zc706
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.host)
    A_ = s.to(A, target.xcel)
    s.partition(A_, hcl.Partition.Block, dim=1, factor=2)
    target.config(compile="vivado_hls", mode="csim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,10))
    np_B = np.zeros((10,10))

    hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(8))
    hcl_B = hcl.asarray(np_B, dtype=hcl.UInt(8))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

if __name__ == "__main__":
    partition_test()