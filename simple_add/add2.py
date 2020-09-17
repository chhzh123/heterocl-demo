import sys
import heterocl as hcl
import numpy as np

def add():
    qtype = hcl.Fixed(16,12)
    A = hcl.placeholder((10,), "A", dtype=qtype)
    def kernel(A):
        return hcl.compute((10,), lambda x: A[x] + 1, "B", dtype=qtype)
    s = hcl.create_schedule(A, kernel)
    target = hcl.platform.aws_f1
    # target.config(compile="vivado_hls", mode="csim")
    # target.config(compile="vivado_hls", mode="debug")
    # target.config(compile="vitis", mode="hw_sim", backend="vhls")
    target.config(compile="vitis", mode="debug", backend="vhls")
    s.to(A, target.xcel)
    s.to(kernel.B,target.host)
    f = hcl.build(s, target=target)
    print(f)

def add_exe():
    qtype = hcl.UInt(1)
    A = hcl.placeholder((10,10), "A", dtype=qtype)
    def kernel(A):
        return hcl.compute((10,10), lambda x, y: A[x][y] | A[x][y], "B", dtype=qtype)
    s = hcl.create_schedule(A, kernel)
    target = hcl.platform.aws_f1
    # target.config(compile="vivado_hls", mode="csim")
    # target.config(compile="vivado_hls", mode="debug")
    target.config(compile="vitis", mode="hw_exe", backend="vhls")
    # target.config(compile="vitis", mode="debug", backend="vhls")
    s.to(A, target.xcel)
    s.to(kernel.B,target.host)
    f = hcl.build(s, target=target)
    np_A = np.random.random((10,10))
    np_B = np.zeros((10,10))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)

if __name__ == "__main__":
    add()
    add_exe()