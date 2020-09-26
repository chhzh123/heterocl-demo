import heterocl as hcl
import numpy as np
 
def test_inter_stage():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, 
                lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, 
                lambda i, j: C[i][j], "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.C, s[kernel.D], depth=10)
    f = hcl.build(s, target)
    np_A = np.zeros((10, 32))
    np_B = np.zeros((10, 32))
    np_D = np.zeros((10, 32))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_D = hcl.asarray(np_D)
    f(hcl_A, hcl_B, hcl_D)

def test_simple_reuse():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        return B

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.B, target.host)
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_B = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)

def test_residual():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i], "C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csim")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel) # off-chip A -> on-chip A'
    s.to(kernel.C, target.host)
    # s.to(A, s[kernel.B]) # on-chip A' -> on-chip B
    s.to(kernel.B, s[kernel.C])
    # s.to(A, s[kernel.C])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

def test_consecutive():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + 1, "C")
        D = hcl.compute(C.shape,
                lambda i: C[i] + 1, "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn|cosim")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C, s[kernel.D])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

def test_zero():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i: A[i] + 1, "B")
        C1 = hcl.compute(A.shape, lambda i: 0, "C1")
        C2 = hcl.compute(A.shape, lambda i: 0, "C2")
        def foo(i):
            C1[i] = B[i] + 1
            C2[i] = C1[i] + 1
        hcl.mutate((10,), lambda i: foo(i), "C")
        D = hcl.compute(A.shape, lambda i: C2[i] + 1, "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csim")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C.C2, s[kernel.D])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_D = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_D = hcl.asarray(np_D)
    f(hcl_A, hcl_D)

def test_duplicated():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + 1, "C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])
    s.to(kernel.B, s[kernel.C]) # duplicated streaming
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

def test_residual2():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + 1, "C")
        D = hcl.compute(C.shape,
                lambda i: B[i] + C[i], "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.B, s[kernel.C], depth=10)
    s.to(kernel.B, s[kernel.D], depth=10)
    s.to(kernel.C, s[kernel.D], depth=10)
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

def test_complex():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: hcl.select(i < 9, B[i] + B[i+1], B[i]),"C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

def test_hierarchy():
    dtype = hcl.Float()
    A = hcl.placeholder((4, 4), "A", dtype)

    def kernel(A):

        def func(data):
            out = hcl.compute((4, 4),lambda x, y: 0, "out", dtype)
            with hcl.Stage("S"):
                with hcl.for_(0, 4, name="i") as i:
                    with hcl.for_(0, 4, name="j") as j:
                        out[i, j] = data[i, j] + 1
            return out

        B = func(A)
        C = hcl.compute((4,4), lambda i, j: B[i, j] + 1, "C")
        return C

    s = hcl.create_schedule([A], kernel)

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn")
    s.to(A, target.xcel)
    # s.to(kernel.out, target.xcel)
    s.to(kernel.C, target.host)
    f = hcl.build(s, target=target)
    np_A = np.random.randint(0, 10, A.shape)
    hcl_A = hcl.asarray(np_A,dtype)
    hcl_B = hcl.asarray(np.zeros((4, 4),np.float),dtype)
    f(hcl_A, hcl_B)

def test_imperative():
    dtype = hcl.Float()
    A = hcl.placeholder((4, 4), "A", dtype)

    def kernel(A):

        def pool(data):
            out = hcl.compute((2, 2),lambda x, y: 0, "out", dtype)
            with hcl.Stage("S"):
                LB = hcl.compute((2, 4),lambda x, y: 0, "LB", dtype)
                with hcl.for_(0, 2, name="x") as x:
                    with hcl.for_(0, 2, name="y") as y:
                        with hcl.for_(0, 2, name="LB_i") as LB_i:
                            with hcl.for_(0, 4, name="LB_j") as LB_j:
                                LB[LB_i, LB_j] = data[x * 2 + LB_i, LB_j]
                        val = hcl.scalar(0,"val")
                        with hcl.for_(0, 2, name="r") as r:
                            with hcl.for_(0, 2, name="c") as c:
                                val.v += LB[r, y * 2 + c]
                        out[x, y] = val / 4
            return out

        B = hcl.compute((4,4), lambda i, j: A[i, j] + 1, "B")
        C = pool(B)
        D = hcl.compute((2,2), lambda i, j: C[i, j] + 1, "D")
        return C

    s = hcl.create_schedule([A], kernel)
    s[kernel.S].pipeline(kernel.S.axis[1])
    s.partition(A, dim=2)

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="pool.prj")
    s.to(A, target.xcel)
    s.to(kernel.S.B, target.host)
    f = hcl.build(s, target=target)
    np_A = np.random.randint(0, 10, A.shape)
    hcl_A = hcl.asarray(np_A,dtype)
    hcl_B = hcl.asarray(np.zeros((2, 2),np.float),dtype)
    f(hcl_A, hcl_B)

from collections import OrderedDict
import heterocl.tvm as tvm

def test_dataflow():
    A = hcl.placeholder((1,10), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i, j: A[i, j] + 1, "B", attrs=OrderedDict([('app',tvm.make.StringImm('B'))]))
        C = hcl.compute(B.shape,
                lambda i, j: B[i, j] + 1, "C", attrs=OrderedDict([('app',tvm.make.StringImm('C'))]))
        D = hcl.compute(C.shape,
                lambda i, j: C[i, j] + 1, "D", attrs=OrderedDict([('app',tvm.make.StringImm('D'))]))
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C, s[kernel.D])
    f = hcl.build(s, target)
    np_A = np.zeros((1,10))
    np_D = np.zeros((1,10))
    hcl_A = hcl.asarray(np_A)
    hcl_D = hcl.asarray(np_D)
    f(hcl_A, hcl_D)

def test_interface():
    A = hcl.placeholder((10,), "A")
    W = hcl.placeholder((10,), "W")

    def kernel(A, W):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + W[i], "C")
        return C

    target = hcl.platform.aws_f1
    target.config(compile="vitis", mode="hw_exe", project="project-vitis.prj")
    s = hcl.create_schedule([A, W], kernel)
    s.to([A, W], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_W = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_W = hcl.asarray(np_W)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_W, hcl_C)

if __name__ == "__main__":
    # test_inter_stage()
    # test_simple_reuse()
    # test_residual()
    # test_consecutive()
    test_zero()
    # test_duplicated()
    # test_residual2()
    # test_complex()
    # test_imperative()
    # test_hierarchy()
    # test_dataflow()
    # test_interface()