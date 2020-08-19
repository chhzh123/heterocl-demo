import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report
import hlib

def simple_add():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode):
        A = hcl.placeholder((10, 32), "A", dtype=hcl.UInt(8))
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=hcl.UInt(8))
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "B", dtype=hcl.UInt(8))
            D = hcl.compute(A.shape, lambda *args : C[args] + 1, "D", dtype=hcl.UInt(8))
            return D
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.D, target.host)
        target.config(compile="vivado_hls", mode=target_mode)
        # sys.exit()
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(8))
        hcl_B = hcl.asarray(np_B, dtype=hcl.UInt(8))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode:
            report = f.report("csyn")
            assert "ReportVersion" in report
        elif "csim" in target_mode:
            for i in range(0, 10):
                for j in range(0, 32):
                    assert ret_B[i, j] == (np_A[i, j] + 3)

    # test_hls("csim")
    test_hls("csyn")
    # test_hls("csim|csyn")
    # test_hls("cosim")

def one_conv():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    A = hcl.placeholder((1,1,16,16), "A")
    w1 = hcl.placeholder((1,3,3,3), "w1")

    def kernel(A, w1):
        conv1 = hlib.op.nn.conv2d_nchw(A, w1, padding=[1,1], name="conv1")
        return conv1
    
    target = hcl.platform.zc706
    s = hcl.create_schedule([A, w1], kernel)
    s.to([A, w1], target.xcel)
    s.to(kernel.conv1, target.host)
    target.config(compile="vivado_hls", mode="csim")
    f = hcl.build(s, target)

    np_A = np.random.randint(0, 256, size=(1,1,16,16))
    np_w1 = np.random.randint(0, 10, size=(1,3,3,3))
    np_B = np.zeros((1,3,16,16))

    hcl_A = hcl.asarray(np_A)
    hcl_w1 = hcl.asarray(np_w1)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_w1, hcl_B)

def double_conv():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    A = hcl.placeholder((1,1,16,16), "A")
    w1 = hcl.placeholder((1,3,3,3), "w1")
    w2 = hcl.placeholder((3,6,3,3), "w2")

    def kernel(A, w1, w2):
        conv1 = hlib.op.nn.conv2d_nchw(A, w1, padding=[1,1], name="conv1")
        conv2 = hlib.op.nn.conv2d_nchw(conv1, w2, padding=[1,1], name="conv2")
        return conv2
    
    target = hcl.platform.zc706
    s = hcl.create_schedule([A, w1, w2], kernel)
    s.to([A, w1, w2], target.xcel)
    s.to(kernel.conv2, target.host)
    target.config(compile="vivado_hls", mode="csyn")
    f = hcl.build(s, target)

    np_A = np.random.randint(0, 256, size=(1,1,16,16))
    np_w1 = np.random.randint(0, 10, size=(1,3,3,3))
    np_w2 = np.random.randint(0, 10, size=(3,6,3,3))
    np_B = np.zeros((1,6,16,16))

    hcl_A = hcl.asarray(np_A)
    hcl_w1 = hcl.asarray(np_w1)
    hcl_w2 = hcl.asarray(np_w2)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_w1, hcl_w2, hcl_B)

def simple_add2():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    dtype = hcl.Fixed(16,12)
    # dtype = hcl.Float()

    def test_hls(target_mode):
        A = hcl.placeholder((10, 32), "A", dtype=dtype)
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=dtype)
            C = hcl.compute(B.shape, lambda *args : B[args] + 1, "C", dtype=dtype)
            return C
        
        target = hcl.platform.zc706
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.C, target.host)
        s.to(kernel.B, s[kernel.C])
        target.config(compile="vivado_hls", mode=target_mode)
        # sys.exit()
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A, dtype=dtype)
        hcl_B = hcl.asarray(np_B, dtype=dtype)
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode:
            report = f.report("csyn")
            assert "ReportVersion" in report
        elif "csim" in target_mode:
            for i in range(0, 10):
                for j in range(0, 32):
                    assert ret_B[i, j] == (np_A[i, j] + 3)

    test_hls("csim")

if __name__ == '__main__':
    # simple_add()
    # one_conv()
    # double_conv()
    simple_add2()