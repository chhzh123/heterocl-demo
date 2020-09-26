import heterocl as hcl
import numpy as np
import hlib.op.bnn as bnn

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
    s.partition(kernel.F, hcl.Partition.Cyclic, factor=KERNEL_SIZE)
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
    F_ = s.partition(F, hcl.Partition.Cyclic, factor=KERNEL_SIZE, dim=2)
    s.partition(F_, hcl.Partition.Cyclic, factor=KERNEL_SIZE, dim=1)
    # s.partition(F, hcl.Partition.Cyclic, factor=3, dim=2)
    # s.partition(F, hcl.Partition.Cyclic, factor=3, dim=1)
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
    target.config(compile="vivado_hls",mode="csyn|cosim",project="conv3")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_B = hcl.asarray(np.zeros((SIZE-KERNEL_SIZE+1, SIZE-KERNEL_SIZE+1)))
    f(hcl_A, hcl_B)

def conv4():
    dtype = hcl.UInt(4)
    A = hcl.placeholder((1, 1, 8, 8), name="A", dtype=dtype)
    F = hcl.placeholder((16, 1, 3, 3), name="F", dtype=dtype)

    def kernel(A, F):
        return bnn.conv2d_nchw(A, F, padding=[1,1], name="conv1", out_dtype=hcl.UInt(16))

    s = hcl.create_schedule([A, F], kernel)
    s_conv = kernel.conv1
    LB = s.reuse_at(kernel.conv1_pad._op, s[s_conv], s_conv.axis[2], "LB")
    WB = s.reuse_at(LB, s[s_conv], s_conv.axis[3], "WB")
    s[kernel.B].pipeline(kernel.B.axis[2])

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csim|csyn|cosim",project="conv")
    s.to([A, F], target.xcel)
    s.to(kernel.conv1, target.host)
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape), dtype)
    hcl_F = hcl.asarray(np.random.randint(0, 10, F.shape), dtype)
    hcl_B = hcl.asarray(np.zeros((1,16,8,8)), hcl.UInt(16))
    f(hcl_A, hcl_F, hcl_B)

def conv5():
    dtype = hcl.UInt(4)
    A = hcl.placeholder((1, 1, 3, 3), name="A", dtype=dtype)
    F = hcl.placeholder((2, 1, 3, 3), name="F", dtype=dtype)

    def kernel(A, F):
        return bnn.conv2d_nchw(A, F, padding=[1,1], name="conv1", out_dtype=hcl.UInt(16))

    s = hcl.create_schedule([A, F], kernel)
    s_conv = kernel.conv1
    LB = s.reuse_at(kernel.conv1_pad._op, s[s_conv], s_conv.axis[2], "LB")
    WB = s.reuse_at(LB, s[s_conv], s_conv.axis[3], "WB")
    s[kernel.conv1].pipeline(kernel.conv1.axis[2])

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csim|csyn|cosim",project="conv5")
    s.to([A, F], target.xcel)
    s.to(kernel.conv1, target.host)
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape), dtype)
    hcl_F = hcl.asarray(np.random.randint(0, 10, F.shape), dtype)
    hcl_B = hcl.asarray(np.zeros((1,2,3,3)), hcl.UInt(16))
    f(hcl_A, hcl_F, hcl_B)

if __name__ == "__main__":
    # conv1()
    # conv2()
    # conv3()
    # conv4()
    conv5()