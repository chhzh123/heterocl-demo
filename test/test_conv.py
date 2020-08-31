import heterocl as hcl
import numpy as np

hcl.init()

def conv1():
    A = hcl.placeholder((6, 6), "A")

    def kernel(A):
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        F = hcl.copy(np.random.randint(0,10,(3,3)),"F")
        return hcl.compute((4, 4),
                lambda y, x: hcl.sum(A[y+r, x+c] * F[r, c], axis=[r, c]), "B")

    s = hcl.create_schedule([A], kernel)
    LB = s.reuse_at(A, s[kernel.B], kernel.B.axis[0], "LB")
    WB = s.reuse_at(LB, s[kernel.B], kernel.B.axis[1], "WB")
    # s.partition(LB, dim=1)
    # s.partition(WB)
    s.partition(kernel.F)
    s[kernel.B].pipeline(kernel.B.axis[1])

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="conv1")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_B = hcl.asarray(np.zeros((4, 4)))
    f(hcl_A, hcl_B)

def conv2():
    A = hcl.placeholder((6, 6), "A")
    F = hcl.placeholder((3, 3), "F")

    def kernel(A, F):
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        return hcl.compute((4, 4),
                lambda y, x: hcl.sum(A[y+r, x+c] * F[r, c], axis=[r, c]), "B")

    s = hcl.create_schedule([A, F], kernel)
    LB = s.reuse_at(A, s[kernel.B], kernel.B.axis[0], "LB")
    WB = s.reuse_at(LB, s[kernel.B], kernel.B.axis[1], "WB")
    # s.partition(LB, dim=1)
    # s.partition(WB)
    s.partition(F)
    s[kernel.B].pipeline(kernel.B.axis[1])
    
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="conv2")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_F = hcl.asarray(np.random.randint(0, 10, F.shape))
    hcl_B = hcl.asarray(np.zeros((4, 4)))
    f(hcl_A, hcl_F, hcl_B)

def conv3():
    A = hcl.placeholder((6, 6), "A")

    def kernel(A):
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        F = hcl.const_tensor(np.random.randint(0,10,(3,3)), "F")
        return hcl.compute((4, 4),
                lambda y, x: hcl.sum(A[y+r, x+c] * F[r, c], axis=[r, c]), "B")

    s = hcl.create_schedule([A], kernel)
    LB = s.reuse_at(A, s[kernel.B], kernel.B.axis[0], "LB")
    WB = s.reuse_at(LB, s[kernel.B], kernel.B.axis[1], "WB")
    # s.partition(LB, dim=1)
    # s.partition(WB)
    # s.partition(kernel.F)
    s[kernel.B].pipeline(kernel.B.axis[1])

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="conv3")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_B = hcl.asarray(np.zeros((4, 4)))
    f(hcl_A, hcl_B)

if __name__ == "__main__":
    conv1()
    conv2()
    conv3()