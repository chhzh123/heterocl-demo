import heterocl as hcl
import numpy as np
import sys
from functools import partial


def systolic(m=2, k=2, n=2, w=2, h=2, dtype=hcl.Int(), target=None):

    hcl.init(dtype)

    Ain = hcl.placeholder((m, k), dtype=dtype, name="A")
    Bin = hcl.placeholder((k, n), dtype=dtype, name="B")
    # Output = hcl.placeholder((m, n), dtype=dtype, name="output")

    def systolic_array(A, B):

        # define modules with loop
        @hcl.def_([(1,), (1,), ()])
        def pe(a, b, x):
            with hcl.if_(x == 0):
                result = a * b
                hcl.return_(a)
            with hcl.elif_(x == 1):
                hcl.return_(b)
            with hcl.else_():
                hcl.return_(result)

        # PE = {f'pe_{i}' : partial(pe) for i in range(w*h)}
        PE = {}
        for i in range(w * h):
            with hcl.Stage("pe_{}".format(i)):
                PE['pe_{}'.format(i)] = partial(pe)

        # each k calls of update function calculate one block of result matrix
        # b_row: block row index
        # b_col: block col index
        def update(b_row, b_col, k, O):
            # fetch input
            localA = []
            localB = []
            for input_a in range(h):
                localA.append(hcl.compute((1,), lambda x : A[input_a + h * b_row, k], "localA_{}".format(input_a)))
            for input_b in range(w):
                localB.append(hcl.compute((1,), lambda x : B[k, input_b + w * b_col], "localB_{}".format(input_b)))

            # systolic connection
            net = [[None] * h] * w
            for i in range(h + w - 1):
                for row in range(i + 1):
                    col = i - row
                    if col < 0 or col > w-1 or row > h-1: continue
                    ## instantiate a PE and record partial results
                    input_a = localA[row] if col == 0 else hcl.compute((1,), lambda x : net[row][col-1][0], "input_a{}{}".format(row, col))
                    input_b = localB[col] if row == 0 else hcl.compute((1,), lambda x : net[row-1][col][1], "input_b{}{}".format(row, col))
                    out = hcl.compute((3,), lambda x : PE['pe_%d' % (row * w + col)](
                        input_a, input_b, x), "out_{}{}".format(row, col))
                    O[row + h * b_row, col + w * b_col] += out[2]
                    net[row][col] = out

        block_rows = int(m / h)
        block_cols = int(n / w)
        O = hcl.compute((m, n), lambda *args : 0, name="Output")
        hcl.mutate((block_rows, block_cols, k), lambda b_row, b_col, k: update(b_row, b_col, k, O), name="update")
        return O

    s = hcl.create_schedule([Ain, Bin], systolic_array)

    # data_graph = s.dataflow_graph(plot=True)

    print(hcl.lower(s))

    # pipeline loop
    k = systolic_array.update
    s[k].pipeline(k.axis[0])
    s[k].pipeline(k.axis[1])
    s[k].pipeline(k.axis[2])
    print(hcl.build(s, target="vhls"))
    sys.exit()

    # systolic connection with .to()
    # s.to(k.input_a01, s[PE['pe_0']], s[PE['pe_1']])
    # s.to(systolic_array.Output, systolic_array.update) # self loopback

    if isinstance(target, hcl.platform):
        s.to([Ain, Bin], target.xcel)
        s.to(systolic_array.Output, target.host)
        target.config(compile="vivado_hls", mode="csyn")

    return hcl.build(s, target=target)void default_function(ap_int<32> A[2][2], ap_int<32> B[2][2], ap_int<32> Output[2][2]) {
  ap_int<32> _top;
  ap_int<32> pe;
  ap_int<32> pe_0;
  ap_int<32> pe_1;
  ap_int<32> pe_2;
  ap_int<32> pe_3;
  for (ap_int<32> args = 0; args < 2; ++args) {
    for (ap_int<32> args0 = 0; args0 < 2; ++args0) {
      Output[args][args0] = 0;
    }
  }
  ap_int<32> update;
  for (ap_int<32> b_row = 0; b_row < 1; ++b_row) {
  #pragma HLS pipeline
    for (ap_int<32> b_col = 0; b_col < 1; ++b_col) {
    #pragma HLS pipeline
      for (ap_int<32> k = 0; k < 2; ++k) {
      #pragma HLS pipeline
        ap_int<32> localA_0;
        localA_0 = A[(b_row * 2)][k];
        ap_int<32> localA_1;
        localA_1 = A[((b_row * 2) + 1)][k];
        ap_int<32> localB_0;
        localB_0 = B[(b_col + k)][0];
        ap_int<32> localB_1;
        localB_1 = B[(b_col + k)][1];
        ap_int<32> out_00[3];
        for (ap_int<32> x = 0; x < 3; ++x) {
          out_00[x] = pe(localA_0, localB_0, x);
        }
        Output[(b_col + (b_row * 2))][0] = ((ap_int<32>)(((ap_int<33>)Output[(b_col + (b_row * 2))][0]) + ((ap_int<33>)out_00[2])));
        ap_int<32> input_a01;
        input_a01 = out_00[0];
        ap_int<32> out_01[3];
        for (ap_int<32> x1 = 0; x1 < 3; ++x1) {
          out_01[x1] = pe(input_a01, localB_1, x1);
        }
        Output[(b_col + (b_row * 2))][1] = ((ap_int<32>)(((ap_int<33>)Output[(b_col + (b_row * 2))][1]) + ((ap_int<33>)out_01[2])));
        ap_int<32> input_b10;
        input_b10 = out_00[1];
        ap_int<32> out_10[3];
        for (ap_int<32> x2 = 0; x2 < 3; ++x2) {
          out_10[x2] = pe(localA_1, input_b10, x2);
        }
        Output[((b_col + (b_row * 2)) + 1)][0] = ((ap_int<32>)(((ap_int<33>)Output[((b_col + (b_row * 2)) + 1)][0]) + ((ap_int<33>)out_10[2])));
        ap_int<32> input_a11;
        input_a11 = out_10[0];
        ap_int<32> input_b11;
        input_b11 = out_01[1];
        ap_int<32> out_11[3];
        for (ap_int<32> x3 = 0; x3 < 3; ++x3) {
          out_11[x3] = pe(input_a11, input_b11, x3);
        }
        Output[((b_col + (b_row * 2)) + 1)][1] = ((ap_int<32>)(((ap_int<33>)Output[((b_col + (b_row * 2)) + 1)][1]) + ((ap_int<33>)out_11[2])));
      }
    }
  }
}


# matrix size
# m*k k*n
m = 2
k = 2
n = 2
# systolic size
w = 2
h = 2

np_1 = np.random.randint(10, size=(m,k))
np_2 = np.random.randint(10, size=(k,n))

dtype = hcl.Int()

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m,n)), dtype=dtype)

# systolic array
target = hcl.platform.zc706
fs = systolic(m, k, n, w, h, dtype=hcl.Int(), target=target)
fs(hcl_m1, hcl_m2, hcl_m3)
print("Systolic Array's result = ")
print(hcl_m3.asnumpy())

answer = np.dot(np_1, np_2)
print("Correct Answer = ")
print(answer)


if np.array_equal(hcl_m3.asnumpy(), answer):
    print("Yeah we got that right")
else:
    print("And I Oop...")
