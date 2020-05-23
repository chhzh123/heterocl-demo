import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report

def test_llvm():

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = None # hcl.platform.llvm
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()
    np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

if __name__ == "__main__":
    test_llvm()