import heterocl as hcl
import numpy as np

A = hcl.placeholder((8, 8), "A")
B = hcl.placeholder((8, 8), "B", dtype=hcl.Fixed(16,12))
def kernel(A, B):
    return hcl.compute((8, 8), lambda y, x: 
        hcl.select(x < 4, A[y][x], B[y][x]), "C")
s = hcl.create_scheme([A, B], kernel)
s = hcl.create_schedule_from_scheme(s)
f = hcl.build(s, target="vhls")
print(f)