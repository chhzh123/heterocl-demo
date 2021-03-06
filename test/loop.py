import heterocl as hcl
import os, sys
import numpy as np

size = 2

def test_loop():

    hcl.init()
    A = hcl.placeholder((size, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        return B
    
    target = None # hcl.platform.zc706
    s = hcl.create_schedule([A], kernel)
    s_B = kernel.B
    s[s_B].pipeline(s_B.axis[0])
    f = hcl.build(s, target)
    print(hcl.lower(s))

if __name__ == "__main__":
    test_loop()