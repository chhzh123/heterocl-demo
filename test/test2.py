import heterocl as hcl
import numpy as np

def test():
    dtype = hcl.Fixed(12,10)
    def kernel():
        A = hcl.const_tensor(np.random.random((10,10)), "A", dtype)
        return hcl.compute(A.shape, lambda x, y: A[x, y]+1, "B", dtype)
    s = hcl.create_schedule([], kernel)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn")
    f = hcl.build(s, target=target)
    hcl_B = hcl.asarray(np.zeros((10,10)))
    f(hcl_B)

import heterocl.tvm as tvm

def test_complex():
    dtype = hcl.Float()
    A = hcl.placeholder((1,1,8,8),"A",dtype)
    def kernel(A):
        return hcl.compute((1,1,10,10), lambda i, c, x, y: hcl.select(tvm.all(x < 8, y < 8),A[i, c, x, y],0), "B", dtype)
        # return hcl.compute((1,1,10,10), lambda i, c, x, y: A[i, c, x, y], "B", dtype)
    s = hcl.create_schedule([A], kernel)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csim")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.zeros((1,1,8,8)))
    hcl_B = hcl.asarray(np.zeros((1,1,10,10)))
    f(hcl_A,hcl_B)

if __name__ == "__main__":
    # test()
    test_complex()