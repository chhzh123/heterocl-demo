import sys
import heterocl as hcl
import numpy as np

A = hcl.placeholder((64,), "A")

def kernel(A):
    B = hcl.compute((64,), lambda x: A[x], "B")
    C = hcl.compute((8, 8), lambda x, y: A[x * 8 + y] + B[x * 8 + y], "C")
    return C

s = hcl.create_schedule([A], kernel)
kernel_B = kernel.B
kernel_C = kernel.C
x_out, x_in = s[kernel_B].split(kernel_B.axis[0],8)
s[kernel_B].compute_at(s[kernel_C], kernel_C.axis[1])
print(hcl.lower(s))