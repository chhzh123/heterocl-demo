# http://heterocl.csl.cornell.edu/doc/tutorials/tutorial_08_backend.html

import sys
import heterocl as hcl
import numpy as np

A = hcl.placeholder((10, 10), "A")
def kernel(A):
    return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
s = hcl.create_scheme(A, kernel)
s.downsize(kernel.B, hcl.UInt(4))
s = hcl.create_schedule_from_scheme(s)
s.partition(A)
s[kernel.B].pipeline(kernel.B.axis[1])

if sys.argv[1] == "llvm":
	f = hcl.build(s, target="llvm")
	hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
	hcl_B = hcl.asarray(np.zeros((8, 8)), dtype=hcl.UInt(4))
	f(hcl_A, hcl_B)
	np_A = hcl_A.asnumpy().astype(np.int32)
	np_B = (np_A[:8,:8] + np_A[2:,2:])[:8,:8]
	print(np_A)
	print(hcl_A)
	print(np_B)
	print(hcl_B)
else:
	f = hcl.build(s, target="vhls")
	with open("simple_add.cpp","w") as outfile:
		outfile.write(f)