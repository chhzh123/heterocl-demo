import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode):
        A = hcl.placeholder((10, 32), "A", dtype=hcl.UInt(8))
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=hcl.UInt(8))
            return B
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.B, target.host)
        target.config(compile="vivado_hls", mode=target_mode)
        # sys.exit()
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(8))
        hcl_B = hcl.asarray(np_B, dtype=hcl.UInt(8))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode:
            report = f.report("csyn")
            assert "ReportVersion" in report
        elif "csim" in target_mode:
            for i in range(0, 10):
                for j in range(0, 32):
                    assert ret_B[i, j] == (np_A[i, j] + 1)

    # test_hls("csim")
    test_hls("csyn")
    # test_hls("csim|csyn")
    # test_hls("cosim")

if __name__ == '__main__':
    test_vivado_hls()
    # report.parse_xml("project",True)