import heterocl as hcl
import numpy as np

def test():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x][y] + 1, "C")
        return C
    s = hcl.create_schedule(A, kernel)
    s[kernel.B].compute_at(s[kernel.C], kernel.C.axis[0])
    s[kernel.C].dataflow(kernel.C.axis[0])
    f = hcl.build(s,"vhls")
    print(f)

def test2():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        return B
    s = hcl.create_schedule(A, kernel)
    s[kernel.B].dataflow(kernel.B.axis[0])
    f = hcl.build(s,"vhls")
    print(f)

def test3():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x][y] + 1, "C")
        return C
    s = hcl.create_schedule(A, kernel)
    s[kernel.B].pipeline(kernel.B.axis[0])
    s[kernel.C].pipeline(kernel.C.axis[0])
    s.dataflow()
    f = hcl.build(s,"vhls")
    print(f)

def test_csyn():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x][y] + 1, "C")
        return C
    s = hcl.create_schedule(A, kernel)
    s.partition(A)
    kernel_B = kernel.B
    s[kernel_B].dataflow(kernel_B.axis[0])
    s[kernel_B].pipeline(kernel_B.axis[1])
    # s.dataflow()
    print("====================================")
    # print(hcl.lower(s))
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="debug")
    s.to(A,target.xcel)
    s.to(kernel.C,target.host)
    f = hcl.build(s,target)
    print(f)

if __name__ == "__main__":
    test()
    # test2()
    # test3()
    # test_csyn()