import heterocl as hcl
import numpy as np

A = hcl.placeholder((8, 8), "A")
B = hcl.placeholder((8, 8), "B")
def kernel(A, B):
    C = hcl.compute((8, 8), lambda y, x: A[y][x] + 1, name="add")
    D = hcl.compute((8, 8), lambda y, x: B[y][x] + 1, name="add")
    E = hcl.compute((8, 8), lambda y, x: A[y][x] + B[y][x], name="E")
    return E

s = hcl.create_schedule([A, B], kernel)
print(kernel.__dict__.keys())
s_add = kernel.add
s[s_add].reorder(s_add.axis[1],s_add.axis[0])
print(hcl.lower(s))