import heterocl as hcl
import numpy as np

m = 64
n = 64
k = 64
dtype = hcl.Int()

def GEMM():
    matrix_1 = hcl.placeholder((m, k), dtype=dtype)
    matrix_2 = hcl.placeholder((k, n), dtype=dtype)

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n),
                lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y],
                                     axis=r, dtype=dtype),
                dtype=dtype,
                name="out_matrix")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    block_size = 8
    # y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    # x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    # s[out_matrix].reorder(y0, x0, y1, x1)
    s[out_matrix].pipeline(out_matrix.axis[1])
    target = hcl.platform.zc706
    s.to([matrix_1, matrix_2], target.xcel)
    s.to(kernel.out_matrix, target.host)
    target.config(compile="vivado_hls", mode="csyn")

    # prepare the input data and output placeholder to store the result
    hcl_m1 = hcl.asarray(np.random.randint(10, size=(m, k)), dtype=dtype)
    hcl_m2 = hcl.asarray(np.random.randint(10, size=(k, n)), dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

    f = hcl.build(s, target)
    f(hcl_m1, hcl_m2, hcl_m3)
    report = f.report()

if __name__ == "__main__":
    GEMM()