import heterocl as hcl
import numpy as np

def foo(x):
    x -= ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x += (x >> 8)
    x += (x >> 16)
    return x & 0x0000003F

def test():
    hcl.init()
    A = hcl.placeholder((8, 8), "A")
    def kernel(A):
        return hcl.compute((8, 8), lambda y, x: 
            foo(A[y, x] + A[y, x]), "C")
    s = hcl.create_scheme([A], kernel)
    s = hcl.create_schedule_from_scheme(s)
    f = hcl.build(s,"vhls")
    print(f)

def popcnt(num):
    out = hcl.scalar(0, "popcnt")
    with hcl.for_(0, 8) as i:
        out.v += num[i]
    return out.v

def test2():
    hcl.init()
    A = hcl.placeholder((3,), dtype=hcl.UInt(8), name="A")
    B = hcl.placeholder((3,), dtype=hcl.UInt(8), name="B")
    rb = hcl.reduce_axis(0, 8, name="rb")
    out = hcl.compute((3,), lambda x: 
                        # popcnt(A[x] ^ B[x]))
                        hcl.sum((A[x] ^ B[x])[rb],axis=rb))
    s = hcl.create_schedule([A, B, out])
    f = hcl.build(s,"vhls")
    print(f)

if __name__ == '__main__':
    # test()
    # for i in range(32):
    #     print(foo(i),end=" ")
    # print()
    test2()