import heterocl as hcl
import numpy as np

def test_tutorial():
    hcl.init()

    A = hcl.placeholder((6, 6), "A")
    F = hcl.placeholder((3, 3), "F")

    def kernel(A, F):
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        return hcl.compute((4, 4),
                lambda y, x: hcl.sum(A[y+r, x+c]*F[r, c], axis=[r, c]), "B")

    # s = hcl.create_schedule([A, F], kernel)
    # print(hcl.lower(s))

    s_x = hcl.create_schedule([A, F], kernel)
    WB = s_x.reuse_at(A, s_x[kernel.B], kernel.B.axis[1], "WB")
    print(hcl.lower(s_x))

def test_reuse_blur_x():
    hcl.init()
    A = hcl.placeholder((10, 10),name="A")
    B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    # print(s[B].op.body)
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_with_to():
    hcl.init()
    A = hcl.placeholder((10, 10), name="A")
    def kernel(A):
        B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2],name="B")
        C = hcl.compute((10, 8), lambda y, x: B[y, x], name="C")
        return C
    s = hcl.create_schedule([A], kernel)
    kernel_B = kernel.B
    RB = s.reuse_at(A, s[kernel_B], kernel_B.axis[1])
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csim")
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    # target = None
    f = hcl.build(s, target)

def test_reuse_compute():
    hcl.init()
    A = hcl.placeholder((10, 10),name="A")
    B = hcl.compute((10, 10), lambda y, x: A[y, x], "B")
    C = hcl.compute((10, 8), lambda y, x: B[y, x] + B[y, x+1] + B[y, x+2], "C")
    s = hcl.create_schedule([A, B, C])
    RB = s.reuse_at(B, s[C], C.axis[1])
    print(hcl.lower(s))
    f = hcl.build(s)

def test_reuse_compute_nd():
    hcl.init()
    nz = 1
    rx = hcl.reduce_axis(0, 3, name="rx")
    rz = hcl.reduce_axis(0, nz, name="rz")
    A = hcl.placeholder((nz, 10, 10),name="A")
    B = hcl.compute((10, 8), lambda y, x: hcl.sum(A[rz, y, x+rx],axis=[rz, rx]), "B")
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    print(hcl.lower(s))
    f = hcl.build(s)

def test_reuse_compute_sum():
    hcl.init()
    rx = hcl.reduce_axis(0, 3, name="rx")
    A = hcl.placeholder((10, 10),name="A")
    B = hcl.compute((10, 10), lambda y, x: A[y, x], "B")
    C = hcl.compute((10, 8), lambda y, x: hcl.sum(B[y, x+rx],axis=rx), "C")
    s = hcl.create_schedule([A, B, C])
    RB = s.reuse_at(B, s[C], C.axis[1])
    print(hcl.lower(s))
    f = hcl.build(s)

def test_reuse_compute2():
    hcl.init()
    A = hcl.placeholder((10, 10),name="A")
    def kernel(A):
        B = hcl.compute((10, 10), lambda y, x: A[y, x], "B")
        C = hcl.compute((10, 8), lambda y, x: B[y, x] + B[y, x+1] + B[y, x+2], "C")
        return C
    s = hcl.create_schedule([A], kernel)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csim")
    B_ = s.to(kernel.B, target.xcel)
    C_ = s.to(kernel.C, target.host)
    RB = s.reuse_at(B_, s[kernel.C], kernel.C.axis[1])
    print(hcl.lower(s))
    f = hcl.build(s, target)

# test_reuse_blur_x()
# test_tutorial()
# test_reuse_compute()
# test_reuse_compute_sum()
# test_reuse_compute_nd()
test_reuse_compute2()
# test_reuse_blur_x_with_to()

# hcl_Bxy = hcl.asarray(np.zeros((4, 4)))
# f = hcl.build(s_xy)
# f(hcl_A, hcl_F, hcl_Bxy)
# print('Output without reuse buffers:')
# print(hcl_B)
# print('Output with reuse buffers:')
# print(hcl_Bxy)

# s_final = hcl.create_schedule([A, F], kernel)
# LB = s_final.reuse_at(A, s_final[kernel.B], kernel.B.axis[0], "LB")
# WB = s_final.reuse_at(LB, s_final[kernel.B], kernel.B.axis[1], "WB")
# s_final.partition(LB, dim=1)
# s_final.partition(WB)
# s_final.partition(F)
# s_final[kernel.B].pipeline(kernel.B.axis[1])
# print(hcl.lower(s_final))

# f = hcl.build(s_final, target="vhls")
# print(f)