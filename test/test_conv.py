import heterocl as hcl
import numpy as np

SIZE = 6
KERNEL_SIZE = 3
hcl.init()

def conv1():
    A = hcl.placeholder((SIZE, SIZE), "A")

    def kernel(A):
        r = hcl.reduce_axis(0, KERNEL_SIZE)
        c = hcl.reduce_axis(0, KERNEL_SIZE)
        F = hcl.copy(np.random.randint(0,10,(KERNEL_SIZE,KERNEL_SIZE)),"F")
        return hcl.compute((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1),
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
    hcl_B = hcl.asarray(np.zeros((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1)))
    f(hcl_A, hcl_B)

def conv2():
    A = hcl.placeholder((SIZE, SIZE), "A")
    F = hcl.placeholder((KERNEL_SIZE, KERNEL_SIZE), "F")

    def kernel(A, F):
        r = hcl.reduce_axis(0, KERNEL_SIZE)
        c = hcl.reduce_axis(0, KERNEL_SIZE)
        return hcl.compute((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1),
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
    hcl_B = hcl.asarray(np.zeros((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1)))
    f(hcl_A, hcl_F, hcl_B)

def conv3():
    A = hcl.placeholder((SIZE, SIZE), "A")

    def kernel(A):
        r = hcl.reduce_axis(0, KERNEL_SIZE)
        c = hcl.reduce_axis(0, KERNEL_SIZE)
        F = hcl.const_tensor(np.random.randint(0,10,(KERNEL_SIZE,KERNEL_SIZE)), "F")
        return hcl.compute((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1),
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
    hcl_B = hcl.asarray(np.zeros((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1)))
    f(hcl_A, hcl_B)

if __name__ == "__main__":
    conv1()
    conv2()
    conv3()