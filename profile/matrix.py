import heterocl as hcl
import os, sys
import numpy as np
from heterocl.profiler import Profiler

target = "vhls"
profiler = Profiler(op=[hcl.OpType.Add,hcl.OpType.Mul])

def test(): # 0.25
    dtype = hcl.UInt(16)
    A = hcl.placeholder((10, 32), "A", dtype=dtype)
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x, y] + 1, "B", dtype=dtype)
        return B
    
    s = hcl.create_schedule([A], kernel)
    # target = hcl.platform.zc706
    # s.to(kernel.B,target.xcel)
    # s.to(kernel.C,target.host)
    # target.config(compile="vivado_hls", mode="csim")
    hcl.lower(s, profiler=profiler)
    # f = hcl.build(s, target)
    # print(f)

def gemm(): # 166
    dtype = hcl.Float()
    M = 64
    K = 64
    N = 64
    A = hcl.placeholder((M, K), "A", dtype=dtype)
    B = hcl.placeholder((K, N), "B", dtype=dtype)
    k = hcl.reduce_axis(0, K)
    def kernel(A, B):
        C = hcl.compute((M, N), lambda x, y: hcl.sum(A[x, k] * B[k, y], axis=k, dtype=dtype), "C", dtype=dtype)
        return C
    
    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn", project="gemm")

    def make_schedule(opt=False):
        s = hcl.create_schedule([A, B], kernel)
        s.to([A, B],target.xcel)
        s.to(kernel.C,target.host)

        def optimization():
            s[kernel.C].pipeline(kernel.C.axis[1])
            s.partition(A,hcl.Partition.Block,dim=2,factor=16)
            s.partition(B,hcl.Partition.Block,dim=1,factor=16)

        if opt:
            optimization()
        f = hcl.build(s, target, profiler=profiler)

        np_A = np.random.randint(0, 10, (M, K))
        np_B = np.random.randint(0, 10, (K, N))
        np_C = np.zeros((M, N))
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np_C)
        f(hcl_A, hcl_B, hcl_C)

    make_schedule(opt=False)
    make_schedule(opt=True)
    profiler.roofline(filename="gemm-roofline.png")

def mv_mul(): # 0.5
    dtype = hcl.Float()
    M = 1000
    K = 1000
    A = hcl.placeholder((M, K), "A", dtype=dtype)
    B = hcl.placeholder((K,), "B", dtype=dtype)
    k = hcl.reduce_axis(0, K)
    def kernel(A, B):
        C = hcl.compute((M,), lambda x: hcl.sum(A[x, k] * B[k], axis=k, dtype=dtype), "C", dtype=dtype)
        return C

    s = hcl.create_schedule([A, B], kernel)
    # target = hcl.platform.zc706
    # s.to(kernel.B,target.xcel)
    # s.to(kernel.C,target.host)
    # target.config(compile="vivado_hls", mode="csim")
    hcl.lower(s, profiler=profiler)

if __name__ == "__main__":
    # test()
    gemm()
    # mv_mul()