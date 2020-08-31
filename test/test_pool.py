import heterocl as hcl
import numpy as np
import torch
import torch.nn as nn

dtype = hcl.Float()
hcl.init(dtype)
sum = hcl.reducer(0, lambda x, y: x + y, dtype)

def pool():
    A = hcl.placeholder((4, 4), "A", dtype)

    def kernel(A):
        r = hcl.reduce_axis(0, 2)
        c = hcl.reduce_axis(0, 2)
        return hcl.compute((2, 2),
                lambda x, y: sum(A[x * 2 + r, y * 2 + c], axis=[r, c]) / 4, "B", dtype)

    s = hcl.create_schedule([A], kernel)
    s[kernel.B].pipeline(kernel.B.axis[1])
    s.partition(A, dim=2)

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="pool.prj")
    # target = None
    f = hcl.build(s, target=target)
    np_A = np.random.randint(0, 10, A.shape)
    hcl_A = hcl.asarray(np_A,dtype)
    hcl_B = hcl.asarray(np.zeros((2, 2),np.float),dtype)
    f(hcl_A, hcl_B)
    avgpool = nn.AvgPool2d((2,2))
    np_A = np_A[np.newaxis, np.newaxis, :]
    np_out = avgpool(torch.Tensor(np_A))
    hcl_out = hcl_B.asnumpy()[np.newaxis, np.newaxis, :]
    np.testing.assert_array_equal(np_out,hcl_out)

def pool2():
    A = hcl.placeholder((4, 4), "A", dtype)
    def kernel(A):
        B = hcl.compute((2, 2),lambda x, y: 0, "B", dtype) # syntax sugar
        with hcl.Stage("S"):
            LB = hcl.compute((2, 4),lambda x, y: 0, "LB", dtype)
            with hcl.for_(0, 2, name="x") as x:
                with hcl.for_(0, 2, name="y") as y:
                    with hcl.for_(0, 2, name="LB_i") as LB_i:
                        with hcl.for_(0, 4, name="LB_j") as LB_j:
                            LB[LB_i, LB_j] = A[x * 2 + LB_i, LB_j]
                    val = hcl.scalar(0,"val")
                    with hcl.for_(0, 2, name="r") as r:
                        with hcl.for_(0, 2, name="c") as c:
                            val.v += LB[r, y * 2 + c]
                    B[x, y] = val / 4
        return B

    s = hcl.create_schedule([A], kernel)
    s[kernel.S].pipeline(kernel.S.axis[1])
    s.partition(A, dim=2)

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn",project="pool.prj")
    s.to(A, target.xcel)
    s.to(kernel.S.B, target.host) # here!
    # target = None
    # target="vhls"
    f = hcl.build(s, target=target)
    print(f)
    np_A = np.random.randint(0, 10, A.shape)
    hcl_A = hcl.asarray(np_A,dtype)
    hcl_B = hcl.asarray(np.zeros((2, 2),np.float),dtype)
    f(hcl_A, hcl_B)
    avgpool = nn.AvgPool2d((2,2))
    np_A = np_A[np.newaxis, np.newaxis, :]
    np_out = avgpool(torch.Tensor(np_A))
    hcl_out = hcl_B.asnumpy()[np.newaxis, np.newaxis, :]
    np.testing.assert_array_equal(np_out,hcl_out)
    print(np_out,hcl_out)

if __name__ == "__main__":
    pool()
    pool2()