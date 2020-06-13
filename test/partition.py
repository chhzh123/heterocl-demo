import heterocl as hcl
import numpy as np

def test():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x][y] + 1, "C")
        return C
    s = hcl.create_schedule(A, kernel)
    s.partition(kernel.C)
    print(hcl.lower(s))

if __name__ == "__main__":
    test()