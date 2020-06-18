import heterocl as hcl
import os, sys
import numpy as np
from heterocl.profiler import Profiler

target = "vhls"
profiler = Profiler()

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
    hcl.lower(s)
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
    
    s = hcl.create_schedule([A, B], kernel)
    # target = hcl.platform.zc706
    # s.to(kernel.B,target.xcel)
    # s.to(kernel.C,target.host)
    # target.config(compile="vivado_hls", mode="csim")
    hcl.lower(s, profiler=profiler)
    profiler.roofline()
    # print(hcl.lower(s))
    # f = hcl.build(s, target)
    # print(f)

def mv_mul():
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
    hcl.lower(s)

if __name__ == "__main__":
    # test()
    gemm()
    # mv_mul()