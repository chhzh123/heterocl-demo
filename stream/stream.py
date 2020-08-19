import heterocl as hcl
import numpy as np

# hcl.init()
# target = hcl.platform.zc706
# initiation_interval = 4

# a = hcl.placeholder((10, 20), name="a")
# b = hcl.placeholder((10, 20), name="b")
# c = hcl.placeholder((10, 20), name="c") 
# d = hcl.placeholder((10, 20), name="d")
# e = hcl.placeholder((10, 20), name="e")

# def add_mul(a, b, c, d, e):
#     @hcl.def_([a.shape, b.shape, c.shape])
#     def ret_add(a, b, c):
#         with hcl.for_(0, a.shape[0]) as i:
#             with hcl.for_(0, a.shape[1]) as j:
#                 c[i, j] = a[i, j] + b[i, j]

#     @hcl.def_([c.shape, d.shape, e.shape])
#     def ret_mul(c, d, e):
#         # hcl.update(c, lambda x, y: a[x, y] * b[x, y], 'c_mul')
#         with hcl.for_(0, c.shape[0]) as i:
#             with hcl.for_(0, c.shape[1]) as j:
#                 e[i, j] = c[i, j] * d[i, j]

#     ret_add(a, b, c)
#     ret_mul(c, d, e)

# # compute customization
# s = hcl.create_schedule([a, b, c, d, e], add_mul)
# # op1 = add_mul.ret_add.c
# # op2 = add_mul.ret_mul.c
# # s[op1].pipeline(op1.axis[0], initiation_interval)

# # stream into modules / device
# a0, b0 = s.to([a, b], target.xcel)
# d0 = s.to(d, target.xcel)
# #s.partition(b0, dim=2, factor=2)
# s.to([a0, b0], s[add_mul.ret_add])
# s.to(d0, s[add_mul.ret_mul])

# # within device move producer to consumer
# s.to(c, s[add_mul.ret_mul],
#         s[add_mul.ret_add], depth=10)

# # return tensor for inter-device move
# # e0 = s.stream_to(e, hcl.CPU('riscv'))

# # print(add_mul.ret_mul._buf, c._buf)
# print(hcl.lower(s))
# code = hcl.build(s, target)
# print(code)
# # 
# # with open("example.cl", "w") as f:
# #   f.write(code)
# #   f.close()
 
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
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel) # off-chip A -> on-chip A'
    s.to(kernel.C, target.host)
    s.to(A, s[kernel.B]) # on-chip A' -> on-chip B
    # s.to(kernel.B, s[kernel.C])
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
                lambda i: B[i], "C")
        D = hcl.compute(C.shape,
                lambda i: C[i], "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.B, s[kernel.C], depth=10)
    s.to(kernel.B, s[kernel.D], depth=10)
    # s.to(kernel.C, s[kernel.D], depth=10)
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

if __name__ == "__main__":
    # test_inter_stage()
    # test_simple_reuse()
    test_consecutive()
    test_residual2()