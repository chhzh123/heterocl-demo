import heterocl as hcl
from itertools import permutations
import os
import numpy as np

def test_vivado_hls():
    # if os.system("which vivado_hls >> /dev/null") != 0:
    #     return 

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.zc706
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    target.config(compile="vivado_hls", mode="sw_sim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    print(f)
    ret_B = hcl_B.asnumpy()
    print(ret_B)
    print((np_A+2)*2)

    for i in range(0, 10):
      for j in range(0, 32):
        assert ret_B[i, j] == (np_A[i, j] + 2) *2

if __name__ == '__main__':
    test_vivado_hls()