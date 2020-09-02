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
    target.config(compile="vivado_hls", mode="csyn")
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

if __name__ == "__main__":
    # test_inter_stage()
    # test_simple_reuse()
    # test_residual()
    # test_consecutive()
    # test_duplicated()
    # test_residual2()
    # test_complex()
    # test_imperative()
    test_hierarchy()