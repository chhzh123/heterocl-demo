import heterocl as hcl
import os, sys
import numpy as np

def test_subgraph():
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        return hcl.compute((10,), lambda x: A[x] + 1, "B")
    s = hcl.create_schedule(A, kernel)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csim")
    A_ = s.to(A, target.xcel)
    B_ = s.to(kernel.B,target.host)
    nodes = s.subgraph(inputs=[A_], outputs=[B_])
    print(nodes)
    f = hcl.build(s, target=target)

if __name__ == "__main__":
    test_subgraph()