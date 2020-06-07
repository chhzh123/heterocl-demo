import heterocl as hcl
import numpy as np

def test1():
	A = hcl.placeholder((8, 8), "A")
	B = hcl.placeholder((8, 8), "B", dtype=hcl.Fixed(16,12))
	def kernel(A, B):
	    return hcl.compute((8, 8), lambda y, x: 
	        hcl.select(x < 4, A[y][x], B[y][x]), "C")
	s = hcl.create_scheme([A, B], kernel)
	s = hcl.create_schedule_from_scheme(s)
	f = hcl.build(s, target="vhls")
	print(f)

def test2():
	A = hcl.placeholder((8, 8), "A", dtype=hcl.UInt(1))
	def kernel(A):
	    return hcl.compute((8, 8), lambda y, x: 
	        hcl.select(x < 4, A[y][x], 0), "B")
	s = hcl.create_scheme([A], kernel)
	s = hcl.create_schedule_from_scheme(s)
	f = hcl.build(s, target="vhls")
	with open("select_test.cpp","w") as outfile:
		outfile.write(f)

def test3():
	A = hcl.placeholder((8, 8), "A", dtype=hcl.UInt(2))
	B = hcl.placeholder((8, 8), "B", dtype=hcl.UInt(2))
	def kernel(A, B):
	    return hcl.compute((8, 8), lambda y, x: 
	        hcl.select(x < 4, A[y, x][0], 0), "C")
	s = hcl.create_scheme([A, B], kernel)
	s = hcl.create_schedule_from_scheme(s)
	f = hcl.build(s, "vhls")
	print(f)

if __name__ == '__main__':
	# test1()
	# test2()
	test3()