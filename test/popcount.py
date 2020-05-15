import heterocl as hcl
import numpy as np
import heterocl.tvm as tvm

def test1():
    hcl.init()
    a = hcl.placeholder((3,), dtype=hcl.UInt(8), name="a")
    out = hcl.compute((3,),
        lambda x: tvm.intrin.popcount(a[x]),
        dtype=hcl.UInt(32))
    s = hcl.create_schedule([a, out])
    f = hcl.build(s)
    hcl_a = hcl.asarray(np.array([9, 7, 31]), dtype=hcl.UInt(8))
    hcl_out = hcl.asarray(np.array([0, 0, 0]), dtype=hcl.UInt(32))
    f(hcl_a, hcl_out)
    print("Input : {}".format(hcl_a.asnumpy()))
    print("Output : {}".format(hcl_out.asnumpy()))

# def test2():
A = hcl.placeholder((3,), dtype=hcl.UInt(2), name="A")
for op1 in ["|","&"]:
    for op2 in ["|","&"]:
        for val1 in [1,2]:
            for val2 in [1,2]:
                test_case = "(A[i] {} {}) {} {}".format(op1,val1,op2,val2)
                try:
                    def kernel(A):
                        return hcl.compute((3,), lambda i: eval(test_case), dtype=hcl.UInt(2))
                    s = hcl.create_schedule([A], kernel)
                    f = hcl.build(s)
                    print("{} passed".format(test_case))
                except:
                    print("{} failed".format(test_case))

# A = hcl.placeholder((3,), dtype=hcl.UInt(2), name="A")
# def kernel(A):
#     out = hcl.compute((3,), lambda i: eval("A[i] & 2 | 1"), dtype=hcl.UInt(2))
#     return out
# s = hcl.create_schedule([A], kernel)
# f = hcl.build(s)
# hcl_a = hcl.asarray(np.array([0, 1, 2]), dtype=hcl.UInt(2))
# hcl_out = hcl.asarray(np.array([0, 0, 0]), dtype=hcl.UInt(2))
# f(hcl_a, hcl_out)
# print("Input : {}".format(hcl_a.asnumpy()))
# print("Output : {}".format(hcl_out.asnumpy()))

# if __name__ == '__main__':
#     test2()