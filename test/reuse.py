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

test_reuse_blur_x()
test_tutorial()

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